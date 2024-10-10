import Mathlib

namespace min_product_of_prime_sum_l2883_288359

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p → 
  m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' → 
    m' + n' = p' → m' * n' * p' ≥ m * n * p) → 
  m * n * p = 30 := by
sorry

end min_product_of_prime_sum_l2883_288359


namespace log_inequality_l2883_288343

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 6 / Real.log 3 →
  b = Real.log 10 / Real.log 5 →
  c = Real.log 14 / Real.log 7 →
  a > b ∧ b > c := by
  sorry

end log_inequality_l2883_288343


namespace parabola_equation_correct_l2883_288329

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := y^2 = -4*x

-- Theorem statement
theorem parabola_equation_correct :
  ∀ x y : ℝ,
  ellipse x y →
  parabola_eq x y →
  (left_focus.1 < 0 ∧ left_focus.2 = 0) →
  (vertex.1 = 0 ∧ vertex.2 = 0) →
  ∃ p : ℝ, p = 2 ∧ y^2 = -2*p*x :=
sorry

end parabola_equation_correct_l2883_288329


namespace sampling_probability_theorem_l2883_288305

/-- Represents the probability of a student being selected in a sampling process -/
def sampling_probability (total_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / total_students

/-- The sampling method described in the problem -/
structure SamplingMethod where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- Theorem stating that the probability of each student being selected is equal and is 25/1002 -/
theorem sampling_probability_theorem (method : SamplingMethod)
  (h1 : method.total_students = 2004)
  (h2 : method.selected_students = 50)
  (h3 : method.eliminated_students = 4) :
  sampling_probability method.total_students method.selected_students = 25 / 1002 :=
sorry

end sampling_probability_theorem_l2883_288305


namespace max_sum_consecutive_integers_l2883_288326

/-- Given consecutive integers x, y, and z satisfying 1/x + 1/y + 1/z > 1/45,
    the maximum value of x + y + z is 402. -/
theorem max_sum_consecutive_integers (x y z : ℤ) :
  (y = x + 1) →
  (z = y + 1) →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z > (1 : ℚ) / 45 →
  ∀ a b c : ℤ, (b = a + 1) → (c = b + 1) →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c > (1 : ℚ) / 45 →
    x + y + z ≥ a + b + c →
  x + y + z = 402 :=
by sorry

end max_sum_consecutive_integers_l2883_288326


namespace apples_in_box_l2883_288368

theorem apples_in_box (initial_apples : ℕ) : 
  (initial_apples / 2 - 25 = 6) → initial_apples = 62 := by
  sorry

end apples_in_box_l2883_288368


namespace joey_sneaker_purchase_l2883_288349

/-- The number of collectible figures Joey needs to sell to buy sneakers -/
def figures_to_sell (sneaker_cost lawn_count lawn_pay job_hours job_pay figure_price : ℕ) : ℕ :=
  let lawn_earnings := lawn_count * lawn_pay
  let job_earnings := job_hours * job_pay
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount + figure_price - 1) / figure_price

theorem joey_sneaker_purchase :
  figures_to_sell 92 3 8 10 5 9 = 2 := by
  sorry

end joey_sneaker_purchase_l2883_288349


namespace richards_day2_distance_l2883_288328

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (day2_distance : ℝ) : Prop :=
  let total_distance : ℝ := 70
  let day1_distance : ℝ := 20
  let day3_distance : ℝ := 10
  let remaining_distance : ℝ := 36
  (day2_distance < day1_distance / 2) ∧
  (day1_distance + day2_distance + day3_distance + remaining_distance = total_distance)

theorem richards_day2_distance :
  ∃ (day2_distance : ℝ), richards_journey day2_distance ∧ day2_distance = 4 := by
  sorry

end richards_day2_distance_l2883_288328


namespace jeff_calculation_correction_l2883_288394

theorem jeff_calculation_correction (incorrect_input : ℕ × ℕ) (incorrect_result : ℕ) 
  (h1 : incorrect_input.1 = 52) 
  (h2 : incorrect_input.2 = 735) 
  (h3 : incorrect_input.1 * incorrect_input.2 = incorrect_result) 
  (h4 : incorrect_result = 38220) : 
  (0.52 : ℝ) * 7.35 = 3.822 := by
sorry

end jeff_calculation_correction_l2883_288394


namespace average_of_a_and_b_l2883_288362

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 17 → 
  b = 2 * a → 
  (a + b) / 2 = 33.5 := by
sorry

end average_of_a_and_b_l2883_288362


namespace cos_double_angle_problem_l2883_288321

theorem cos_double_angle_problem (α : ℝ) (h : Real.cos (π + α) = 2/5) :
  Real.cos (2 * α) = -17/25 := by sorry

end cos_double_angle_problem_l2883_288321


namespace platform_length_l2883_288327

/-- Given a train of length 900 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the length of the platform is 1050 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1050 := by sorry

end platform_length_l2883_288327


namespace animals_on_shore_l2883_288303

/-- Proves that given the initial numbers of animals and the conditions about drowning,
    the total number of animals that made it to shore is 35. -/
theorem animals_on_shore (initial_sheep initial_cows initial_dogs : ℕ)
                         (drowned_sheep : ℕ)
                         (h1 : initial_sheep = 20)
                         (h2 : initial_cows = 10)
                         (h3 : initial_dogs = 14)
                         (h4 : drowned_sheep = 3)
                         (h5 : drowned_sheep * 2 = initial_cows - (initial_cows - drowned_sheep * 2)) :
  initial_sheep - drowned_sheep + (initial_cows - drowned_sheep * 2) + initial_dogs = 35 := by
  sorry

#check animals_on_shore

end animals_on_shore_l2883_288303


namespace train_meeting_distance_l2883_288399

/-- Proves that when two trains starting 200 miles apart and traveling towards each other
    at 20 miles per hour each meet, one train will have traveled 100 miles. -/
theorem train_meeting_distance (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (h1 : total_distance = 200)
  (h2 : speed_a = 20)
  (h3 : speed_b = 20) :
  speed_a * (total_distance / (speed_a + speed_b)) = 100 :=
by sorry

end train_meeting_distance_l2883_288399


namespace outfit_count_l2883_288336

def red_shirts : ℕ := 7
def green_shirts : ℕ := 7
def pants : ℕ := 8
def green_hats : ℕ := 10
def red_hats : ℕ := 10
def blue_hats : ℕ := 5

def total_outfits : ℕ := red_shirts * pants * (green_hats + blue_hats) + 
                          green_shirts * pants * (red_hats + blue_hats)

theorem outfit_count : total_outfits = 1680 := by
  sorry

end outfit_count_l2883_288336


namespace minimum_fourth_exam_score_l2883_288342

def exam1 : ℕ := 86
def exam2 : ℕ := 82
def exam3 : ℕ := 89
def required_increase : ℚ := 2

def average (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4

theorem minimum_fourth_exam_score :
  ∀ x : ℕ,
    (average exam1 exam2 exam3 x ≥ (exam1 + exam2 + exam3 : ℚ) / 3 + required_increase) ↔
    x ≥ 94 :=
by sorry

end minimum_fourth_exam_score_l2883_288342


namespace expression_value_l2883_288348

theorem expression_value : (2023 : ℚ) / 2022 - 2022 / 2023 + 1 = 4098551 / (2022 * 2023) := by
  sorry

end expression_value_l2883_288348


namespace correct_mark_calculation_l2883_288300

/-- Proves that if a mark of 83 in a class of 26 pupils increases the class average by 0.5,
    then the correct mark should have been 70. -/
theorem correct_mark_calculation (total_marks : ℝ) (wrong_mark correct_mark : ℝ) : 
  (wrong_mark = 83) →
  (((total_marks + wrong_mark) / 26) = ((total_marks + correct_mark) / 26 + 0.5)) →
  (correct_mark = 70) := by
sorry

end correct_mark_calculation_l2883_288300


namespace xyz_value_l2883_288312

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = -10 := by
  sorry

end xyz_value_l2883_288312


namespace infinitely_many_triples_divisible_by_p_cubed_l2883_288393

theorem infinitely_many_triples_divisible_by_p_cubed :
  ∀ n : ℕ, ∃ p a b : ℕ,
    p > n ∧
    Nat.Prime p ∧
    a < p ∧
    b < p ∧
    (p^3 : ℕ) ∣ ((a + b)^p - a^p - b^p) :=
by sorry

end infinitely_many_triples_divisible_by_p_cubed_l2883_288393


namespace morios_current_age_l2883_288391

/-- Calculates Morio's current age given the ages of Teresa and Morio at different points in time. -/
theorem morios_current_age
  (teresa_current_age : ℕ)
  (morio_age_at_michikos_birth : ℕ)
  (teresa_age_at_michikos_birth : ℕ)
  (h1 : teresa_current_age = 59)
  (h2 : morio_age_at_michikos_birth = 38)
  (h3 : teresa_age_at_michikos_birth = 26) :
  morio_age_at_michikos_birth + (teresa_current_age - teresa_age_at_michikos_birth) = 71 :=
by sorry

end morios_current_age_l2883_288391


namespace abs_is_even_and_decreasing_l2883_288334

def f (x : ℝ) := abs x

theorem abs_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end abs_is_even_and_decreasing_l2883_288334


namespace mystery_books_ratio_l2883_288345

def total_books : ℕ := 46
def top_section_books : ℕ := 12 + 8 + 4
def bottom_section_books : ℕ := total_books - top_section_books
def known_bottom_books : ℕ := 5 + 6
def mystery_books : ℕ := bottom_section_books - known_bottom_books

theorem mystery_books_ratio :
  (mystery_books : ℚ) / bottom_section_books = 1 / 2 := by
  sorry

end mystery_books_ratio_l2883_288345


namespace fayes_rows_l2883_288375

/-- Given that Faye has 210 crayons in total and places 30 crayons in each row,
    prove that she created 7 rows. -/
theorem fayes_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) :
  total_crayons / crayons_per_row = 7 := by
  sorry

end fayes_rows_l2883_288375


namespace sphere_radius_from_hole_l2883_288351

theorem sphere_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) :
  hole_diameter = 30 →
  hole_depth = 12 →
  sphere_radius = (27 / 8 + 12) →
  sphere_radius = 15.375 := by
sorry

end sphere_radius_from_hole_l2883_288351


namespace tricycle_count_l2883_288382

/-- Represents the number of wheels on a scooter -/
def scooter_wheels : ℕ := 2

/-- Represents the number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := 10

/-- Represents the total number of wheels -/
def total_wheels : ℕ := 26

/-- Theorem stating that the number of tricycles must be 6 given the conditions -/
theorem tricycle_count :
  ∃ (scooters tricycles : ℕ),
    scooters + tricycles = total_vehicles ∧
    scooters * scooter_wheels + tricycles * tricycle_wheels = total_wheels ∧
    tricycles = 6 :=
by sorry

end tricycle_count_l2883_288382


namespace altitude_of_equal_area_triangle_trapezoid_l2883_288367

/-- The altitude of a triangle and trapezoid with equal areas -/
theorem altitude_of_equal_area_triangle_trapezoid
  (h : ℝ) -- altitude
  (b : ℝ) -- base of the triangle
  (m : ℝ) -- median of the trapezoid
  (h_pos : h > 0) -- altitude is positive
  (b_val : b = 24) -- base of triangle is 24 inches
  (m_val : m = b / 2) -- median of trapezoid is half of triangle base
  (area_eq : 1/2 * b * h = m * h) -- areas are equal
  : h ∈ Set.Ioi 0 :=
by sorry

end altitude_of_equal_area_triangle_trapezoid_l2883_288367


namespace prepaid_card_cost_l2883_288387

/-- The cost of a prepaid phone card given call cost, call duration, and remaining balance -/
theorem prepaid_card_cost 
  (cost_per_minute : ℚ) 
  (call_duration : ℕ) 
  (remaining_balance : ℚ) : 
  cost_per_minute = 16/100 →
  call_duration = 22 →
  remaining_balance = 2648/100 →
  remaining_balance + cost_per_minute * call_duration = 30 := by
sorry

end prepaid_card_cost_l2883_288387


namespace min_value_theorem_l2883_288358

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 8 * Real.sqrt 5 := by
  sorry

end min_value_theorem_l2883_288358


namespace potato_bag_weight_l2883_288395

theorem potato_bag_weight (morning_bags : ℕ) (afternoon_bags : ℕ) (total_weight : ℕ) :
  morning_bags = 29 →
  afternoon_bags = 17 →
  total_weight = 322 →
  total_weight / (morning_bags + afternoon_bags) = 7 :=
by sorry

end potato_bag_weight_l2883_288395


namespace contingency_fund_amount_l2883_288347

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def crisis_fund_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def crisis_fund : ℚ := total_donation * crisis_fund_ratio

def remaining_after_main : ℚ := total_donation - community_pantry - crisis_fund
def livelihood_fund : ℚ := remaining_after_main * livelihood_ratio

def contingency_fund : ℚ := remaining_after_main - livelihood_fund

theorem contingency_fund_amount : contingency_fund = 30 := by
  sorry

end contingency_fund_amount_l2883_288347


namespace simplify_fraction_l2883_288380

theorem simplify_fraction : (210 : ℚ) / 315 = 2 / 3 := by
  sorry

end simplify_fraction_l2883_288380


namespace cubic_root_in_interval_l2883_288355

/-- Given a cubic equation with three real roots and a condition on its coefficients,
    prove that at least one root belongs to the interval [0, 2]. -/
theorem cubic_root_in_interval
  (a b c : ℝ)
  (has_three_real_roots : ∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z)
  (coef_sum_bound : 2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r : ℝ, r^3 + a*r^2 + b*r + c = 0 ∧ 0 ≤ r ∧ r ≤ 2 :=
sorry

end cubic_root_in_interval_l2883_288355


namespace soccer_tournament_games_l2883_288320

def soccer_tournament (n : ℕ) (m : ℕ) (tie_breaker : ℕ) : ℕ :=
  let first_stage := n * (n - 1) / 2
  let second_stage := 2 * (m * (m - 1) / 2)
  first_stage + second_stage + tie_breaker

theorem soccer_tournament_games :
  soccer_tournament 25 10 1 = 391 := by
  sorry

end soccer_tournament_games_l2883_288320


namespace q_div_p_equals_225_l2883_288338

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 card of a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 4 * Nat.choose cards_per_number 1 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating that q/p = 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end q_div_p_equals_225_l2883_288338


namespace specific_trapezoid_area_l2883_288379

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longerBase : ℝ
  /-- One of the base angles of the trapezoid -/
  baseAngle : ℝ
  /-- The height of the trapezoid -/
  height : ℝ

/-- The area of the isosceles trapezoid -/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.longerBase = 20 ∧
    t.baseAngle = Real.arcsin 0.6 ∧
    t.height = 9 ∧
    trapezoidArea t = 100 := by
  sorry

end specific_trapezoid_area_l2883_288379


namespace square_perimeter_proof_l2883_288309

theorem square_perimeter_proof (p1 p2 p3 : ℝ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24)
  (h4 : (p3 / 4) ^ 2 = ((p1 / 4) ^ 2) - ((p2 / 4) ^ 2)) : p1 = 40 := by
  sorry

end square_perimeter_proof_l2883_288309


namespace cone_lateral_surface_area_l2883_288373

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) 
  (A : ℝ) :
  r = 3 →
  V = 12 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  A = Real.pi * r * l →
  A = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l2883_288373


namespace sphere_xz_intersection_radius_l2883_288330

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A circle in 3D space -/
structure Circle where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Theorem: The radius of the circle where the sphere intersects the xz-plane is 6 -/
theorem sphere_xz_intersection_radius : 
  ∀ (s : Sphere),
  ∃ (c1 c2 : Circle),
  c1.center = (3, 5, 0) ∧ c1.radius = 3 ∧  -- xy-plane intersection
  c2.center = (0, 5, -6) ∧                 -- xz-plane intersection
  (∃ (x y z : ℝ), s.center = (x, y, z)) →
  c2.radius = 6 := by
sorry


end sphere_xz_intersection_radius_l2883_288330


namespace triangle_angle_bounds_l2883_288371

theorem triangle_angle_bounds (y : ℝ) : 
  y > 0 → 
  y + 10 > y + 5 → 
  y + 10 > 4 * y →
  y + 5 + 4 * y > y + 10 →
  y + 5 + y + 10 > 4 * y →
  4 * y + y + 10 > y + 5 →
  (∃ (p q : ℝ), p < y ∧ y < q ∧ 
    (∀ (p' q' : ℝ), p' < y ∧ y < q' → q' - p' ≥ q - p) ∧
    q - p = 25 / 12) :=
by sorry

end triangle_angle_bounds_l2883_288371


namespace t_minus_s_eq_negative_19_583_l2883_288339

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 6

/-- The list of class enrollments -/
def class_enrollments : List ℕ := [60, 30, 10, 10, 5, 5]

/-- The average number of students per teacher -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- The average number of students per student -/
noncomputable def s : ℚ := (class_enrollments.map (λ x => x * x)).sum / num_students

/-- The difference between t and s -/
theorem t_minus_s_eq_negative_19_583 : t - s = -19583 / 1000 := by sorry

end t_minus_s_eq_negative_19_583_l2883_288339


namespace parabola_focus_l2883_288323

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ (x y : ℝ), y = a * x^2 → (x - f.1)^2 = 4 * (1 / (4 * a)) * (y - f.2) :=
sorry

end parabola_focus_l2883_288323


namespace negation_equivalence_l2883_288333

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 > 2 → |x| > 1 ∨ |y| > 1) ↔ (x^2 + y^2 ≤ 2 → |x| ≤ 1 ∧ |y| ≤ 1) :=
by sorry

end negation_equivalence_l2883_288333


namespace harrison_croissant_expenditure_l2883_288390

/-- The cost of a regular croissant in dollars -/
def regular_croissant_cost : ℚ := 7/2

/-- The cost of an almond croissant in dollars -/
def almond_croissant_cost : ℚ := 11/2

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- The total amount Harrison spends on croissants in a year -/
def total_spent_on_croissants : ℚ := 
  (regular_croissant_cost * weeks_in_year) + (almond_croissant_cost * weeks_in_year)

theorem harrison_croissant_expenditure : 
  total_spent_on_croissants = 468 := by sorry

end harrison_croissant_expenditure_l2883_288390


namespace union_condition_implies_m_leq_4_l2883_288388

/-- Given sets A and B, if their union equals A, then m ≤ 4 -/
theorem union_condition_implies_m_leq_4 (m : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
  let B := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  (A ∪ B = A) → m ≤ 4 := by
sorry

end union_condition_implies_m_leq_4_l2883_288388


namespace zongzi_probability_theorem_l2883_288340

/-- Given a set of 6 items where 2 are of type A and 4 are of type B -/
def total_items : ℕ := 6
def type_A_items : ℕ := 2
def type_B_items : ℕ := 4
def selected_items : ℕ := 3

/-- Probability of selecting at least one item of type A -/
def prob_at_least_one_A : ℚ := 4/5

/-- Probability distribution of X (number of type A items selected) -/
def prob_dist_X : List (ℕ × ℚ) := [(0, 1/5), (1, 3/5), (2, 1/5)]

/-- Mathematical expectation of X -/
def expectation_X : ℚ := 1

/-- Main theorem -/
theorem zongzi_probability_theorem :
  (total_items = type_A_items + type_B_items) →
  (prob_at_least_one_A = 4/5) ∧
  (prob_dist_X = [(0, 1/5), (1, 3/5), (2, 1/5)]) ∧
  (expectation_X = 1) := by
  sorry


end zongzi_probability_theorem_l2883_288340


namespace complex_number_quadrant_l2883_288376

theorem complex_number_quadrant (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) :
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = -1 := by
  sorry

end complex_number_quadrant_l2883_288376


namespace june_election_win_l2883_288378

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (male_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  male_vote_percentage = 675 / 1000 →
  ∃ (female_vote_percentage : ℚ),
    female_vote_percentage = 25 / 100 ∧
    (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
    (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * female_vote_percentage >
    (total_students : ℚ) / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
      (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * x ≤
      (total_students : ℚ) / 2 :=
by sorry

end june_election_win_l2883_288378


namespace markus_marbles_l2883_288365

theorem markus_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_bags = 13 := by
  sorry

end markus_marbles_l2883_288365


namespace iains_old_pennies_l2883_288352

/-- The number of pennies older than Iain -/
def oldPennies : ℕ := 30

/-- The initial number of pennies Iain has -/
def initialPennies : ℕ := 200

/-- The number of pennies Iain has left after removing old pennies and 20% of the remaining -/
def remainingPennies : ℕ := 136

/-- The percentage of remaining pennies Iain throws out -/
def throwOutPercentage : ℚ := 1/5

theorem iains_old_pennies :
  oldPennies = initialPennies - (remainingPennies / (1 - throwOutPercentage)) := by
  sorry

end iains_old_pennies_l2883_288352


namespace sqrt_difference_equals_five_sixths_l2883_288357

theorem sqrt_difference_equals_five_sixths :
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end sqrt_difference_equals_five_sixths_l2883_288357


namespace first_term_formula_l2883_288322

theorem first_term_formula (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) :
  ∃ (rest : ℕ), p^q = (p^(q-1) - p + 1) + rest :=
sorry

end first_term_formula_l2883_288322


namespace perfect_squares_between_200_and_600_l2883_288314

theorem perfect_squares_between_200_and_600 :
  (Finset.filter (fun n => 200 < n^2 ∧ n^2 < 600) (Finset.range 25)).card = 10 :=
by sorry

end perfect_squares_between_200_and_600_l2883_288314


namespace wasted_meat_price_l2883_288307

def minimum_wage : ℝ := 8
def fruit_veg_price : ℝ := 4
def bread_price : ℝ := 1.5
def janitorial_wage : ℝ := 10
def fruit_veg_weight : ℝ := 15
def bread_weight : ℝ := 60
def meat_weight : ℝ := 20
def overtime_hours : ℝ := 10
def james_work_hours : ℝ := 50

theorem wasted_meat_price (meat_price : ℝ) : meat_price = 5 := by
  sorry

end wasted_meat_price_l2883_288307


namespace inequality_proof_l2883_288398

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  a * b > a * c ∧ c * b^2 < a * b^2 := by
  sorry

end inequality_proof_l2883_288398


namespace male_non_listeners_l2883_288350

/-- Radio station survey data -/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem: The number of males who do not listen to the radio station is 105 -/
theorem male_non_listeners (data : SurveyData)
  (h1 : data.total_listeners = 160)
  (h2 : data.total_non_listeners = 200)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 105) :
  data.male_non_listeners = 105 := by
  sorry


end male_non_listeners_l2883_288350


namespace wage_increase_proof_l2883_288316

theorem wage_increase_proof (original_wage new_wage : ℝ) 
  (h1 : new_wage = 70)
  (h2 : new_wage = original_wage * (1 + 0.16666666666666664)) :
  original_wage = 60 := by
  sorry

end wage_increase_proof_l2883_288316


namespace suwy_unique_product_l2883_288392

/-- Represents a letter with its corresponding value -/
structure Letter where
  value : Nat
  h : value ≥ 1 ∧ value ≤ 26

/-- Represents a four-letter list -/
structure FourLetterList where
  letters : Fin 4 → Letter

/-- Calculates the product of a four-letter list -/
def product (list : FourLetterList) : Nat :=
  (list.letters 0).value * (list.letters 1).value * (list.letters 2).value * (list.letters 3).value

theorem suwy_unique_product :
  ∀ (list : FourLetterList),
    product list = 19 * 21 * 23 * 25 →
    (list.letters 0).value = 19 ∧
    (list.letters 1).value = 21 ∧
    (list.letters 2).value = 23 ∧
    (list.letters 3).value = 25 :=
by sorry

end suwy_unique_product_l2883_288392


namespace lesser_fraction_l2883_288324

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end lesser_fraction_l2883_288324


namespace expression_value_l2883_288341

theorem expression_value (a : ℝ) (h1 : 1 < a) (h2 : a < 2) :
  Real.sqrt ((a - 2)^2) + |1 - a| = 1 := by sorry

end expression_value_l2883_288341


namespace inverse_f_sum_l2883_288325

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 2*x - x^2 + 1

theorem inverse_f_sum :
  let f_inv := Function.invFun f
  f_inv (-1) + f_inv 1 + f_inv 5 = 4 + Real.sqrt 3 + Real.sqrt 5 := by
  sorry

end inverse_f_sum_l2883_288325


namespace correct_additional_muffins_l2883_288313

/-- Calculates the additional muffins needed for a charity event -/
def additional_muffins_needed (target : ℕ) (arthur_baked : ℕ) (beatrice_baked : ℕ) (charles_baked : ℕ) : ℕ :=
  target - (arthur_baked + beatrice_baked + charles_baked)

/-- Proves the correctness of additional muffins calculations for three charity events -/
theorem correct_additional_muffins :
  (additional_muffins_needed 200 35 48 29 = 88) ∧
  (additional_muffins_needed 150 20 35 25 = 70) ∧
  (additional_muffins_needed 250 45 60 30 = 115) := by
  sorry

#eval additional_muffins_needed 200 35 48 29
#eval additional_muffins_needed 150 20 35 25
#eval additional_muffins_needed 250 45 60 30

end correct_additional_muffins_l2883_288313


namespace sine_symmetry_l2883_288344

/-- Given a sinusoidal function y = 2sin(3x + φ) with |φ| < π/2,
    if the line of symmetry is x = π/12, then φ = π/4 -/
theorem sine_symmetry (φ : Real) : 
  (|φ| < π/2) →
  (∀ x : Real, 2 * Real.sin (3*x + φ) = 2 * Real.sin (3*(π/6 - x) + φ)) →
  φ = π/4 := by
  sorry

end sine_symmetry_l2883_288344


namespace triangle_side_range_l2883_288381

theorem triangle_side_range :
  ∀ x : ℝ, 
    (∃ t : Set (ℝ × ℝ × ℝ), 
      t.Nonempty ∧ 
      (∀ s ∈ t, s.1 = 3 ∧ s.2.1 = 6 ∧ s.2.2 = x) ∧
      (∀ s ∈ t, s.1 + s.2.1 > s.2.2 ∧ s.1 + s.2.2 > s.2.1 ∧ s.2.1 + s.2.2 > s.1)) →
    3 < x ∧ x < 9 :=
by sorry

end triangle_side_range_l2883_288381


namespace g_difference_theorem_l2883_288366

/-- The function g(x) = 3x^2 + x - 4 -/
def g (x : ℝ) : ℝ := 3 * x^2 + x - 4

/-- Theorem stating that [g(x+h) - g(x)] - [g(x) - g(x-h)] = 6h^2 for all real x and h -/
theorem g_difference_theorem (x h : ℝ) : 
  (g (x + h) - g x) - (g x - g (x - h)) = 6 * h^2 := by
  sorry

end g_difference_theorem_l2883_288366


namespace orange_eating_contest_l2883_288389

theorem orange_eating_contest (num_students : ℕ) (max_oranges min_oranges : ℕ) :
  num_students = 8 →
  max_oranges = 8 →
  min_oranges = 1 →
  max_oranges - min_oranges = 7 := by
sorry

end orange_eating_contest_l2883_288389


namespace b_current_age_l2883_288335

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 5 years older than B.
    Prove that B's current age is 35 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 5) : 
  b = 35 := by
  sorry

end b_current_age_l2883_288335


namespace shirts_per_day_l2883_288304

theorem shirts_per_day (total_shirts : ℕ) (reused_shirts : ℕ) (vacation_days : ℕ) : 
  total_shirts = 11 → reused_shirts = 1 → vacation_days = 7 → 
  (total_shirts - reused_shirts) / (vacation_days - 2) = 2 := by
  sorry

end shirts_per_day_l2883_288304


namespace sandy_molly_age_ratio_l2883_288310

/-- The ratio of Sandy's current age to Molly's current age is 4:3, given that Sandy will be 38 years old in 6 years and Molly is currently 24 years old. -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 38
  let years_until_future : ℕ := 6
  let molly_current_age : ℕ := 24
  let sandy_current_age : ℕ := sandy_future_age - years_until_future
  (sandy_current_age : ℚ) / molly_current_age = 4 / 3 := by
sorry

end sandy_molly_age_ratio_l2883_288310


namespace questionnaire_C_count_l2883_288397

def population : ℕ := 960
def sample_size : ℕ := 32
def first_number : ℕ := 9
def questionnaire_A_upper : ℕ := 450
def questionnaire_B_upper : ℕ := 750

theorem questionnaire_C_count :
  let group_size := population / sample_size
  let groups_AB := questionnaire_B_upper / group_size
  sample_size - groups_AB = 7 := by sorry

end questionnaire_C_count_l2883_288397


namespace x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l2883_288306

theorem x_gt_2_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ 
  ¬(∀ x : ℝ, x^2 > 4 → x > 2) :=
by sorry

end x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l2883_288306


namespace rectangle_perimeter_l2883_288331

/-- Represents the side lengths of the nine squares in the rectangle -/
structure SquareSides where
  a1 : ℕ
  a2 : ℕ
  a3 : ℕ
  a4 : ℕ
  a5 : ℕ
  a6 : ℕ
  a7 : ℕ
  a8 : ℕ
  a9 : ℕ

/-- Checks if the given SquareSides satisfy the conditions of the problem -/
def isValidSquareSides (s : SquareSides) : Prop :=
  s.a1 = 2 ∧
  s.a1 + s.a2 = s.a3 ∧
  s.a1 + s.a3 = s.a4 ∧
  s.a3 + s.a4 = s.a5 ∧
  s.a4 + s.a5 = s.a6 ∧
  s.a2 + s.a3 + s.a5 = s.a7 ∧
  s.a2 + s.a7 = s.a8 ∧
  s.a1 + s.a4 + s.a6 = s.a9 ∧
  s.a6 + s.a9 = s.a7 + s.a8

/-- Represents the dimensions of the rectangle -/
structure RectangleDimensions where
  length : ℕ
  width : ℕ

/-- Checks if the given RectangleDimensions satisfy the conditions of the problem -/
def isValidRectangle (r : RectangleDimensions) : Prop :=
  r.length > r.width ∧
  Even r.length ∧
  Even r.width ∧
  r.length = r.width + 2

theorem rectangle_perimeter (s : SquareSides) (r : RectangleDimensions) :
  isValidSquareSides s → isValidRectangle r →
  r.length = s.a9 → r.width = s.a8 →
  2 * (r.length + r.width) = 68 := by
  sorry


end rectangle_perimeter_l2883_288331


namespace y_order_on_quadratic_l2883_288369

/-- A quadratic function of the form y = x² + 4x + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + k

/-- Theorem stating the order of y-coordinates for specific x-values on the quadratic function -/
theorem y_order_on_quadratic (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : quadratic_function k (-4) = y₁)
  (h₂ : quadratic_function k (-1) = y₂)
  (h₃ : quadratic_function k 1 = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry


end y_order_on_quadratic_l2883_288369


namespace tetrahedron_volume_bound_l2883_288384

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: For any tetrahedron with only one edge length greater than 1, its volume is at most 1/8 -/
theorem tetrahedron_volume_bound (t : Tetrahedron) 
  (h : ∃! i, t.edges i > 1) : volume t ≤ 1/8 := by sorry

end tetrahedron_volume_bound_l2883_288384


namespace billy_sandwiches_l2883_288396

theorem billy_sandwiches (billy katelyn chloe : ℕ) : 
  katelyn = billy + 47 →
  chloe = (katelyn : ℚ) / 4 →
  billy + katelyn + chloe = 169 →
  billy = 49 := by
sorry

end billy_sandwiches_l2883_288396


namespace illuminated_cube_surface_area_l2883_288372

/-- The area of the illuminated part of a cube's surface when a cylindrical beam of light is directed along its main diagonal -/
theorem illuminated_cube_surface_area 
  (a : ℝ) 
  (ρ : ℝ) 
  (h_a : a = 1 / Real.sqrt 2) 
  (h_ρ : ρ = Real.sqrt (2 - Real.sqrt 3)) : 
  ∃ (area : ℝ), area = (Real.sqrt 3 - 3/2) * (Real.pi + 3) := by
  sorry

end illuminated_cube_surface_area_l2883_288372


namespace at_least_one_third_l2883_288377

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end at_least_one_third_l2883_288377


namespace tammy_earnings_l2883_288308

/-- Calculates Tammy's earnings from selling oranges over a period of time. -/
def orange_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℕ) (num_days : ℕ) : ℕ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let total_packs := packs_per_day * num_days
  total_packs * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks equal $840. -/
theorem tammy_earnings : 
  orange_earnings 10 12 6 2 (3 * 7) = 840 := by
  sorry

end tammy_earnings_l2883_288308


namespace jacks_tire_slashing_l2883_288301

theorem jacks_tire_slashing (tire_cost window_cost total_cost : ℕ) 
  (h1 : tire_cost = 250)
  (h2 : window_cost = 700)
  (h3 : total_cost = 1450) :
  ∃ (num_tires : ℕ), num_tires * tire_cost + window_cost = total_cost ∧ num_tires = 3 := by
  sorry

end jacks_tire_slashing_l2883_288301


namespace unique_divisor_problem_l2883_288354

theorem unique_divisor_problem (dividend : Nat) (divisor : Nat) : 
  dividend = 12128316 →
  divisor * 7 < 1000 →
  divisor * 7 ≥ 100 →
  dividend % divisor = 0 →
  (∀ d : Nat, d ≠ divisor → 
    (d * 7 < 1000 ∧ d * 7 ≥ 100 ∧ dividend % d = 0) → False) →
  divisor = 124 := by
sorry

end unique_divisor_problem_l2883_288354


namespace circular_permutations_2a2b2c_l2883_288360

/-- The number of first-type circular permutations for a multiset with given element counts -/
def circularPermutations (counts : List Nat) : Nat :=
  sorry

/-- Theorem: The number of first-type circular permutations for 2 a's, 2 b's, and 2 c's is 16 -/
theorem circular_permutations_2a2b2c :
  circularPermutations [2, 2, 2] = 16 := by
  sorry

end circular_permutations_2a2b2c_l2883_288360


namespace inscribed_cube_volume_l2883_288317

/-- A pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)
  (is_equilateral_base : base_side > 0)
  (is_isosceles_lateral : height^2 + (base_side/2)^2 = (base_side * Real.sqrt 3 / 2)^2)

/-- A cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (touches_base_center : side_length ≤ p.base_side * Real.sqrt 3 / 3)
  (touches_apex : 2 * side_length = p.height)

/-- The volume of the inscribed cube is 1/64 -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) 
    (h1 : p.base_side = 1) : c.side_length^3 = 1/64 := by
  sorry

end inscribed_cube_volume_l2883_288317


namespace max_sin_x_sin_2x_l2883_288302

theorem max_sin_x_sin_2x (x : Real) (h : 0 < x ∧ x < π / 2) :
  (∀ y : Real, y = Real.sin x * Real.sin (2 * x) → y ≤ 4 * Real.sqrt 3 / 9) ∧
  (∃ y : Real, y = Real.sin x * Real.sin (2 * x) ∧ y = 4 * Real.sqrt 3 / 9) :=
sorry

end max_sin_x_sin_2x_l2883_288302


namespace problem_solution_l2883_288363

theorem problem_solution (x y : ℝ) (h1 : 3 * x + 2 = 11) (h2 : y = x - 1) : 6 * y - 3 * x = 3 := by
  sorry

end problem_solution_l2883_288363


namespace village_population_l2883_288386

theorem village_population (P : ℝ) : 
  (P * 1.25 * 0.75 = 18750) → P = 20000 := by
  sorry

end village_population_l2883_288386


namespace area_XYZA_is_four_thirds_l2883_288319

/-- Right trapezoid PQRS with the given properties -/
structure RightTrapezoid where
  PQ : ℝ
  RS : ℝ
  PR : ℝ
  trisectPQ : ℝ → ℝ → ℝ  -- Function to represent trisection points on PQ
  trisectRS : ℝ → ℝ → ℝ  -- Function to represent trisection points on RS
  midpoint : ℝ → ℝ → ℝ   -- Function to calculate midpoint

/-- The area of quadrilateral XYZA in the right trapezoid -/
def areaXYZA (t : RightTrapezoid) : ℝ :=
  let X := t.midpoint 0 (t.trisectPQ 0 1)
  let Y := t.midpoint (t.trisectPQ 0 1) (t.trisectPQ 1 2)
  let Z := t.midpoint (t.trisectRS 1 2) (t.trisectRS 0 1)
  let A := t.midpoint (t.trisectRS 0 1) t.RS
  -- Area calculation would go here
  sorry

/-- Theorem stating that the area of XYZA is 4/3 -/
theorem area_XYZA_is_four_thirds (t : RightTrapezoid) 
    (h1 : t.PQ = 2) 
    (h2 : t.RS = 6) 
    (h3 : t.PR = 4) : 
  areaXYZA t = 4/3 := by
  sorry

end area_XYZA_is_four_thirds_l2883_288319


namespace sleep_increase_l2883_288356

theorem sleep_increase (initial_sleep : ℝ) (increase_fraction : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_fraction = 1/3 →
  final_sleep = initial_sleep + initial_sleep * increase_fraction →
  final_sleep = 8 :=
by
  sorry

end sleep_increase_l2883_288356


namespace words_with_a_count_l2883_288311

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}

def word_length : Nat := 3

def words_with_a (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n - (s.erase 'A').card ^ n

theorem words_with_a_count :
  words_with_a alphabet word_length = 61 := by
  sorry

end words_with_a_count_l2883_288311


namespace day_150_of_previous_year_is_wednesday_l2883_288370

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Function to determine the day of the week for a given day number in a year -/
def dayOfWeek (year : Year) (dayNumber : ℕ) : DayOfWeek :=
  sorry

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem day_150_of_previous_year_is_wednesday 
  (P : Year)
  (h1 : dayOfWeek P 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (Year.mk (P.value + 1)) 300 = DayOfWeek.Friday)
  : dayOfWeek (Year.mk (P.value - 1)) 150 = DayOfWeek.Wednesday :=
sorry

end day_150_of_previous_year_is_wednesday_l2883_288370


namespace two_parts_of_ten_l2883_288383

theorem two_parts_of_ten (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 5 → 
  (x = 7.5 ∧ y = 2.5) ∨ (x = 2.5 ∧ y = 7.5) := by
sorry

end two_parts_of_ten_l2883_288383


namespace parabola_vertex_y_coordinate_l2883_288364

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Define the y-coordinate of the vertex
def vertex_y : ℝ := -3

-- Theorem statement
theorem parabola_vertex_y_coordinate :
  ∃ x : ℝ, ∀ t : ℝ, f t ≥ f x ∧ f x = vertex_y :=
sorry

end parabola_vertex_y_coordinate_l2883_288364


namespace eight_digit_integers_count_l2883_288353

theorem eight_digit_integers_count : 
  (Finset.range 9).card * (Finset.range 10).card ^ 7 = 90000000 := by
  sorry

end eight_digit_integers_count_l2883_288353


namespace john_foundation_homes_l2883_288361

/-- Represents the dimensions of a concrete slab for a home foundation -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the number of homes given foundation parameters -/
def calculateHomes (slab : SlabDimensions) (concreteDensity : Float) (concreteCostPerPound : Float) (totalFoundationCost : Float) : Float :=
  let slabVolume := slab.length * slab.width * slab.height
  let concreteWeight := slabVolume * concreteDensity
  let costPerHome := concreteWeight * concreteCostPerPound
  totalFoundationCost / costPerHome

/-- Proves that John is laying the foundation for 3 homes -/
theorem john_foundation_homes :
  let slab : SlabDimensions := { length := 100, width := 100, height := 0.5 }
  let concreteDensity : Float := 150
  let concreteCostPerPound : Float := 0.02
  let totalFoundationCost : Float := 45000
  calculateHomes slab concreteDensity concreteCostPerPound totalFoundationCost = 3 := by
  sorry


end john_foundation_homes_l2883_288361


namespace greatest_common_factor_of_three_digit_palindromes_l2883_288337

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

def is_valid_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ n = three_digit_palindrome a b

theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧
  (∀ n : ℕ, is_valid_palindrome n → g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n : ℕ, is_valid_palindrome n → d ∣ n) → d ≤ g) ∧
  g = 1 := by sorry

end greatest_common_factor_of_three_digit_palindromes_l2883_288337


namespace remainder_theorem_l2883_288332

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D) :
  P % (D + D') = R :=
sorry

end remainder_theorem_l2883_288332


namespace children_without_candy_l2883_288346

/-- Represents the number of children in the circle -/
def num_children : ℕ := 73

/-- Represents the total number of candies distributed -/
def total_candies : ℕ := 2020

/-- Calculates the position of the nth candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of unique positions reached -/
def unique_positions : ℕ := 37

theorem children_without_candy :
  num_children - unique_positions = 36 :=
sorry

end children_without_candy_l2883_288346


namespace ninth_root_unity_sum_l2883_288385

theorem ninth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (2 * Real.pi * I / 9) →
  z^9 = 1 →
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end ninth_root_unity_sum_l2883_288385


namespace mean_of_six_numbers_with_sum_two_thirds_l2883_288374

theorem mean_of_six_numbers_with_sum_two_thirds :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 2/3 →
  (a + b + c + d + e + f) / 6 = 1/9 := by
sorry

end mean_of_six_numbers_with_sum_two_thirds_l2883_288374


namespace arctan_sum_equals_pi_fourth_l2883_288315

theorem arctan_sum_equals_pi_fourth (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
  sorry

end arctan_sum_equals_pi_fourth_l2883_288315


namespace quadratic_solution_value_l2883_288318

theorem quadratic_solution_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end quadratic_solution_value_l2883_288318
