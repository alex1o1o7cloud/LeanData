import Mathlib

namespace fraction_comparison_l1869_186944

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_comparison_l1869_186944


namespace quadratic_roots_sum_l1869_186900

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end quadratic_roots_sum_l1869_186900


namespace real_part_of_reciprocal_difference_l1869_186910

open Complex

theorem real_part_of_reciprocal_difference (w : ℂ) (h1 : w ≠ 0) (h2 : w.im ≠ 0) (h3 : abs w = 2) :
  (1 / (2 - w)).re = 1/2 := by
  sorry

end real_part_of_reciprocal_difference_l1869_186910


namespace min_value_theorem_l1869_186999

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  (x + 1) * (y + 2) ≥ 15 := by
  sorry

end min_value_theorem_l1869_186999


namespace simplify_expression_l1869_186945

theorem simplify_expression :
  ∀ x y : ℝ, (5 - 6*x) - (9 + 5*x - 2*y) = -4 - 11*x + 2*y :=
by sorry

end simplify_expression_l1869_186945


namespace rectangle_max_area_l1869_186965

/-- Given a square with side length 40 cm that is cut into 5 identical rectangles,
    the length of the shorter side of each rectangle that maximizes its area is 8 cm. -/
theorem rectangle_max_area (square_side : ℝ) (num_rectangles : ℕ) 
  (h1 : square_side = 40)
  (h2 : num_rectangles = 5) :
  let rectangle_area := square_side^2 / num_rectangles
  let shorter_side := square_side / num_rectangles
  shorter_side = 8 ∧ 
  ∀ (w : ℝ), w > 0 → w * (square_side^2 / (num_rectangles * w)) ≤ rectangle_area :=
by sorry

end rectangle_max_area_l1869_186965


namespace minimum_distances_to_pond_l1869_186975

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a walk in cardinal directions -/
inductive Walk
  | North : Nat → Walk
  | South : Nat → Walk
  | East : Nat → Walk
  | West : Nat → Walk

/-- Calculates the end point after a series of walks -/
def end_point (start : Point) (walks : List Walk) : Point :=
  walks.foldl
    (fun p w =>
      match w with
      | Walk.North n => { x := p.x, y := p.y + n }
      | Walk.South n => { x := p.x, y := p.y - n }
      | Walk.East n => { x := p.x + n, y := p.y }
      | Walk.West n => { x := p.x - n, y := p.y })
    start

/-- Calculates the Manhattan distance between two points -/
def manhattan_distance (p1 p2 : Point) : Nat :=
  (p1.x - p2.x).natAbs + (p1.y - p2.y).natAbs

/-- Anička's initial walk -/
def anicka_walk : List Walk :=
  [Walk.North 5, Walk.East 2, Walk.South 3, Walk.West 4]

/-- Vojta's initial walk -/
def vojta_walk : List Walk :=
  [Walk.South 3, Walk.West 4, Walk.North 1]

theorem minimum_distances_to_pond :
  let anicka_start : Point := { x := 0, y := 0 }
  let vojta_start : Point := { x := 0, y := 0 }
  let pond := end_point anicka_start anicka_walk
  let vojta_end := end_point vojta_start vojta_walk
  vojta_end.x + 5 = pond.x →
  manhattan_distance anicka_start pond = 4 ∧
  manhattan_distance vojta_start pond = 3 :=
by sorry


end minimum_distances_to_pond_l1869_186975


namespace art_exhibition_problem_l1869_186987

/-- Art exhibition visitor and ticket problem -/
theorem art_exhibition_problem 
  (total_saturday : ℕ)
  (sunday_morning_increase : ℚ)
  (sunday_afternoon_increase : ℚ)
  (total_sunday_increase : ℕ)
  (sunday_morning_revenue : ℕ)
  (sunday_afternoon_revenue : ℕ)
  (sunday_morning_adults : ℕ)
  (sunday_afternoon_adults : ℕ)
  (h1 : total_saturday = 300)
  (h2 : sunday_morning_increase = 40 / 100)
  (h3 : sunday_afternoon_increase = 30 / 100)
  (h4 : total_sunday_increase = 100)
  (h5 : sunday_morning_revenue = 4200)
  (h6 : sunday_afternoon_revenue = 7200)
  (h7 : sunday_morning_adults = 70)
  (h8 : sunday_afternoon_adults = 100) :
  ∃ (sunday_morning sunday_afternoon adult_price student_price : ℕ),
    sunday_morning = 140 ∧
    sunday_afternoon = 260 ∧
    adult_price = 40 ∧
    student_price = 20 := by
  sorry


end art_exhibition_problem_l1869_186987


namespace sphere_volume_diameter_relation_l1869_186990

theorem sphere_volume_diameter_relation :
  ∀ (V₁ V₂ d₁ d₂ : ℝ),
  V₁ > 0 → d₁ > 0 →
  V₁ = (π * d₁^3) / 6 →
  V₂ = 2 * V₁ →
  V₂ = (π * d₂^3) / 6 →
  d₂ / d₁ = (2 : ℝ)^(1/3) :=
by sorry

end sphere_volume_diameter_relation_l1869_186990


namespace f_pi_third_eq_half_l1869_186958

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi / 2 + α)) / 
  (Real.cos (-Real.pi / 2 + α) * Real.tan (Real.pi + α))

theorem f_pi_third_eq_half : f (Real.pi / 3) = 1 / 2 := by sorry

end f_pi_third_eq_half_l1869_186958


namespace visible_cubes_12_cube_l1869_186911

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Calculates the number of visible unit cubes from a corner of a cube --/
def visible_unit_cubes (c : Cube n) : ℕ :=
  3 * n^2 - 3 * (n - 1) + 1

/-- Theorem stating that for a 12×12×12 cube, the number of visible unit cubes from a corner is 400 --/
theorem visible_cubes_12_cube :
  ∃ (c : Cube 12), visible_unit_cubes c = 400 :=
sorry

end visible_cubes_12_cube_l1869_186911


namespace ant_walk_theorem_l1869_186913

/-- The length of a cube's side in centimeters -/
def cube_side_length : ℝ := 18

/-- The number of cube edges the ant walks along -/
def number_of_edges : ℕ := 5

/-- The distance the ant walks on the cube's surface -/
def ant_walk_distance : ℝ := cube_side_length * number_of_edges

theorem ant_walk_theorem : ant_walk_distance = 90 := by
  sorry

end ant_walk_theorem_l1869_186913


namespace sum_of_coordinates_symmetric_points_l1869_186955

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if A(a, 2) and B(-3, b) are symmetric with respect to the origin, then a + b = 1 -/
theorem sum_of_coordinates_symmetric_points (a b : ℝ) 
  (h : symmetric_wrt_origin a 2 (-3) b) : a + b = 1 := by
  sorry

end sum_of_coordinates_symmetric_points_l1869_186955


namespace complex_equation_solution_l1869_186923

theorem complex_equation_solution (a b c : ℂ) 
  (eq : 3*a + 4*b + 5*c = 0) 
  (norm_a : Complex.abs a = 1) 
  (norm_b : Complex.abs b = 1) 
  (norm_c : Complex.abs c = 1) : 
  a * (b + c) = -3/5 := by sorry

end complex_equation_solution_l1869_186923


namespace number_percentage_equality_l1869_186949

theorem number_percentage_equality (x : ℝ) :
  (40 / 100) * x = (30 / 100) * 50 → x = 37.5 := by
  sorry

end number_percentage_equality_l1869_186949


namespace complex_equation_solution_l1869_186997

theorem complex_equation_solution (z : ℂ) : 
  z + (1 + 2*I) = 10 - 3*I → z = 9 - 5*I :=
by sorry

end complex_equation_solution_l1869_186997


namespace days_not_played_in_june_l1869_186952

/-- The number of days in June. -/
def june_days : ℕ := 30

/-- The number of songs Vivian plays per day. -/
def vivian_songs : ℕ := 10

/-- The number of songs Clara plays per day. -/
def clara_songs : ℕ := vivian_songs - 2

/-- The total number of songs both Vivian and Clara listened to in June. -/
def total_songs : ℕ := 396

/-- The number of days they played songs in June. -/
def days_played : ℕ := total_songs / (vivian_songs + clara_songs)

theorem days_not_played_in_june : june_days - days_played = 8 := by
  sorry

end days_not_played_in_june_l1869_186952


namespace second_exam_score_l1869_186971

theorem second_exam_score (total_marks : ℕ) (num_exams : ℕ) (first_exam_percent : ℚ) 
  (third_exam_marks : ℕ) (overall_average_percent : ℚ) :
  total_marks = 500 →
  num_exams = 3 →
  first_exam_percent = 45 / 100 →
  third_exam_marks = 100 →
  overall_average_percent = 40 / 100 →
  (first_exam_percent * total_marks + (55 / 100) * total_marks + third_exam_marks) / 
    (num_exams * total_marks) = overall_average_percent :=
by sorry

end second_exam_score_l1869_186971


namespace carrie_weeks_to_buy_iphone_l1869_186940

def iphone_cost : ℕ := 1200
def trade_in_value : ℕ := 180
def weekly_earnings : ℕ := 50

def weeks_needed : ℕ :=
  (iphone_cost - trade_in_value + weekly_earnings - 1) / weekly_earnings

theorem carrie_weeks_to_buy_iphone :
  weeks_needed = 21 :=
sorry

end carrie_weeks_to_buy_iphone_l1869_186940


namespace workshop_workers_l1869_186978

/-- Proves that the total number of workers is 12 given the conditions in the problem -/
theorem workshop_workers (total_average : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) 
  (num_technicians : ℕ) (h1 : total_average = 9000) (h2 : tech_average = 12000) 
  (h3 : non_tech_average = 6000) (h4 : num_technicians = 6) : 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    total_average * total_workers = 
      num_technicians * tech_average + (total_workers - num_technicians) * non_tech_average :=
by
  sorry

end workshop_workers_l1869_186978


namespace keanu_fish_spending_l1869_186960

/-- The number of fish Keanu gave to his dog -/
def dog_fish : ℕ := 40

/-- The number of fish Keanu gave to his cat -/
def cat_fish : ℕ := dog_fish / 2

/-- The cost of each fish in dollars -/
def fish_cost : ℕ := 4

/-- The total number of fish Keanu bought -/
def total_fish : ℕ := dog_fish + cat_fish

/-- The total amount Keanu spent on fish in dollars -/
def total_spent : ℕ := total_fish * fish_cost

theorem keanu_fish_spending :
  total_spent = 240 :=
sorry

end keanu_fish_spending_l1869_186960


namespace linear_system_solution_l1869_186954

theorem linear_system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ - 2*x₂ + x₄ = -3 ∧
   3*x₁ - x₂ - 2*x₃ = 1 ∧
   2*x₁ + x₂ - 2*x₃ - x₄ = 4 ∧
   x₁ + 3*x₂ - 2*x₃ - 2*x₄ = 7) →
  (∃ t u : ℝ, x₁ = -3 + 2*x₂ - x₄ ∧
              x₂ = 2 + (2/5)*t + (3/5)*u ∧
              x₃ = t ∧
              x₄ = u) :=
by sorry

end linear_system_solution_l1869_186954


namespace recycling_points_theorem_l1869_186976

/-- Calculates the points earned for recycling paper -/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the conditions, the total points earned is 5 -/
theorem recycling_points_theorem : 
  points_earned 6 28 2 = 5 := by
  sorry

end recycling_points_theorem_l1869_186976


namespace y_intercept_after_translation_intersection_point_l1869_186905

/-- The y-intercept of a line after vertical translation -/
theorem y_intercept_after_translation (m b h : ℝ) :
  let original_line := fun x => m * x + b
  let translated_line := fun x => m * x + (b + h)
  (translated_line 0) = b + h :=
by
  sorry

/-- Proof that the translated line y = 2x - 1 + 3 intersects y-axis at (0, 2) -/
theorem intersection_point :
  let original_line := fun x => 2 * x - 1
  let translated_line := fun x => 2 * x - 1 + 3
  (translated_line 0) = 2 :=
by
  sorry

end y_intercept_after_translation_intersection_point_l1869_186905


namespace faye_coloring_books_l1869_186914

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

theorem faye_coloring_books : 
  let initial_books : ℕ := 34
  let books_bought : ℕ := 48
  let final_books : ℕ := 79
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end faye_coloring_books_l1869_186914


namespace triangle_angle_range_l1869_186959

theorem triangle_angle_range (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_tan : Real.tan B ^ 2 = Real.tan A * Real.tan C) : 
  π / 3 ≤ B ∧ B < π / 2 := by
sorry

end triangle_angle_range_l1869_186959


namespace proposition_relationship_l1869_186998

theorem proposition_relationship (a b : ℝ) : 
  (∀ a b : ℝ, (a > b ∧ a⁻¹ > b⁻¹) → a > 0) ∧ 
  (∃ a b : ℝ, a > 0 ∧ ¬(a > b ∧ a⁻¹ > b⁻¹)) := by
sorry

end proposition_relationship_l1869_186998


namespace necessary_but_not_sufficient_condition_l1869_186984

theorem necessary_but_not_sufficient_condition :
  (∀ x : ℝ, x > 0 → x > -2) ∧ 
  (∃ x : ℝ, x > -2 ∧ x ≤ 0) :=
by sorry

end necessary_but_not_sufficient_condition_l1869_186984


namespace f_positive_range_f_always_negative_l1869_186956

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |2*x - a|

theorem f_positive_range (x : ℝ) : 
  f 3 x > 0 ↔ 1 < x ∧ x < 5/3 := by sorry

theorem f_always_negative (a : ℝ) : 
  (∀ x < 2, f a x < 0) ↔ a ≥ 4 := by sorry

end f_positive_range_f_always_negative_l1869_186956


namespace picnic_group_size_l1869_186903

theorem picnic_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 := by
  sorry

end picnic_group_size_l1869_186903


namespace no_valid_division_l1869_186901

/-- The total weight of all stones -/
def total_weight : ℕ := (77 * 78) / 2

/-- The weight of the heaviest group for a given k -/
def heaviest_group_weight (k : ℕ) : ℕ := 
  (total_weight + k - 1) / k

/-- The number of stones in the heaviest group for a given k -/
def stones_in_heaviest_group (k : ℕ) : ℕ := 
  (heaviest_group_weight k + 76) / 77

/-- The total number of stones in all groups for a given k -/
def total_stones (k : ℕ) : ℕ := 
  k * (stones_in_heaviest_group k + (k - 1) / 2)

/-- The set of possible values for k -/
def possible_k : Finset ℕ := {9, 10, 11, 12}

theorem no_valid_division : 
  ∀ k ∈ possible_k, total_stones k > 77 := by sorry

end no_valid_division_l1869_186901


namespace transportation_charges_calculation_l1869_186989

theorem transportation_charges_calculation (purchase_price repair_cost selling_price : ℕ) 
  (h1 : purchase_price = 14000)
  (h2 : repair_cost = 5000)
  (h3 : selling_price = 30000)
  (h4 : selling_price = (purchase_price + repair_cost + transportation_charges) * 3 / 2) :
  transportation_charges = 1000 :=
by
  sorry

#check transportation_charges_calculation

end transportation_charges_calculation_l1869_186989


namespace work_hours_calculation_l1869_186907

/-- Calculates the required weekly work hours given summer work details and additional earnings needed --/
def required_weekly_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (additional_earnings_needed : ℕ) : ℕ :=
  let hourly_wage := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_hours_needed := additional_earnings_needed / hourly_wage
  total_hours_needed / school_year_weeks

/-- Theorem stating that under given conditions, the required weekly work hours is 16 --/
theorem work_hours_calculation (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (additional_earnings_needed : ℕ) :
  summer_weeks = 10 →
  summer_hours_per_week = 40 →
  summer_earnings = 4000 →
  school_year_weeks = 50 →
  additional_earnings_needed = 8000 →
  required_weekly_hours summer_weeks summer_hours_per_week summer_earnings school_year_weeks additional_earnings_needed = 16 :=
by
  sorry


end work_hours_calculation_l1869_186907


namespace merchant_profit_l1869_186904

theorem merchant_profit (cost_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  markup_percentage = 50 →
  discount_percentage = 20 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price * (1 - discount_percentage / 100)
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by sorry

end merchant_profit_l1869_186904


namespace range_of_a_l1869_186937

-- Define the propositions P and Q
def P (x a : ℝ) : Prop := |x - a| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the negations of P and Q
def not_P (x a : ℝ) : Prop := ¬(P x a)
def not_Q (x : ℝ) : Prop := ¬(Q x)

-- Define the condition that not_P is sufficient but not necessary for not_Q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, not_P x a → not_Q x) ∧ (∃ x, not_Q x ∧ ¬(not_P x a))

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ -1 ≤ a ∧ a ≤ 6 :=
sorry

end range_of_a_l1869_186937


namespace equation_solutions_l1869_186992

theorem equation_solutions :
  (∃ x : ℝ, 2 * (2 - x) - 5 * (2 - x) = 9 ∧ x = 5) ∧
  (∃ x : ℝ, x / 3 - (3 * x - 1) / 6 = 1 ∧ x = -5) := by
  sorry

end equation_solutions_l1869_186992


namespace no_intersection_l1869_186995

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end no_intersection_l1869_186995


namespace abs_x_lt_2_sufficient_not_necessary_for_quadratic_l1869_186928

theorem abs_x_lt_2_sufficient_not_necessary_for_quadratic :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) :=
by sorry

end abs_x_lt_2_sufficient_not_necessary_for_quadratic_l1869_186928


namespace chocolates_distribution_l1869_186957

/-- Given a large box containing small boxes and chocolate bars, 
    calculate the number of chocolate bars in each small box. -/
def chocolates_per_small_box (total_chocolates : ℕ) (num_small_boxes : ℕ) : ℕ :=
  total_chocolates / num_small_boxes

/-- Theorem: In a large box with 15 small boxes and 300 chocolate bars,
    each small box contains 20 chocolate bars. -/
theorem chocolates_distribution :
  chocolates_per_small_box 300 15 = 20 := by
  sorry

end chocolates_distribution_l1869_186957


namespace train_length_l1869_186946

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 250 →
  crossing_time = 36 →
  train_speed * crossing_time - platform_length = 470 := by
  sorry

end train_length_l1869_186946


namespace intercepted_segment_length_l1869_186931

/-- The length of the line segment intercepted by curve C on line l -/
theorem intercepted_segment_length :
  let line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 - 1 = 0}
  let curve_C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}
  let intersection := line_l ∩ curve_C
  ∃ (A B : ℝ × ℝ), A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
by sorry

end intercepted_segment_length_l1869_186931


namespace shift_down_two_units_l1869_186918

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x - 1

def vertical_shift (h : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => h x - shift

theorem shift_down_two_units :
  vertical_shift f 2 = g :=
sorry

end shift_down_two_units_l1869_186918


namespace no_solution_exists_l1869_186921

theorem no_solution_exists : ¬ ∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end no_solution_exists_l1869_186921


namespace unique_solution_quadratic_l1869_186932

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end unique_solution_quadratic_l1869_186932


namespace unique_line_through_point_with_equal_intercepts_l1869_186925

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The point (2, 1) -/
def point : ℝ × ℝ := (2, 1)

/-- A line passes through a given point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- A line has equal intercepts on both axes -/
def equal_intercepts (l : Line) : Prop :=
  ∃ a : ℝ, l.intercept = a ∧ (-l.intercept / l.slope) = a

/-- There exists a unique line passing through (2, 1) with equal intercepts -/
theorem unique_line_through_point_with_equal_intercepts :
  ∃! l : Line, passes_through l point ∧ equal_intercepts l := by
  sorry

end unique_line_through_point_with_equal_intercepts_l1869_186925


namespace order_of_powers_l1869_186939

theorem order_of_powers : 
  let a : ℕ := 2^55
  let b : ℕ := 3^44
  let c : ℕ := 5^33
  let d : ℕ := 6^22
  a < d ∧ d < b ∧ b < c :=
by sorry

end order_of_powers_l1869_186939


namespace shorts_cost_l1869_186993

def football_cost : ℝ := 3.75
def shoes_cost : ℝ := 11.85
def zachary_money : ℝ := 10
def additional_money_needed : ℝ := 8

theorem shorts_cost : 
  ∃ (shorts_price : ℝ), 
    football_cost + shoes_cost + shorts_price = zachary_money + additional_money_needed ∧ 
    shorts_price = 2.40 := by
sorry

end shorts_cost_l1869_186993


namespace complete_work_together_l1869_186919

/-- The number of days it takes for two workers to complete a job together,
    given the number of days it takes each worker to complete the job individually. -/
def days_to_complete_together (days_a days_b : ℚ) : ℚ :=
  1 / (1 / days_a + 1 / days_b)

/-- Theorem stating that if worker A takes 9 days and worker B takes 18 days to complete a job individually,
    then together they will complete the job in 6 days. -/
theorem complete_work_together :
  days_to_complete_together 9 18 = 6 := by
  sorry

end complete_work_together_l1869_186919


namespace ryan_commute_time_l1869_186961

/-- Ryan's weekly commute time calculation -/
theorem ryan_commute_time : 
  let bike_days : ℕ := 1
  let bus_days : ℕ := 3
  let friend_days : ℕ := 1
  let bike_time : ℕ := 30
  let bus_time : ℕ := bike_time + 10
  let friend_time : ℕ := bike_time / 3
  bike_days * bike_time + bus_days * bus_time + friend_days * friend_time = 160 :=
by
  sorry

end ryan_commute_time_l1869_186961


namespace solution_set_equivalence_l1869_186964

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
by sorry

end solution_set_equivalence_l1869_186964


namespace square_difference_division_l1869_186963

theorem square_difference_division : (175^2 - 155^2) / 20 = 330 := by sorry

end square_difference_division_l1869_186963


namespace abc_product_absolute_value_l1869_186962

theorem abc_product_absolute_value (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_eq : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) : 
  |a * b * c| = 1 := by
  sorry

end abc_product_absolute_value_l1869_186962


namespace rectangle_area_invariance_l1869_186947

theorem rectangle_area_invariance (x y : ℝ) :
  (x + 5/2) * (y - 2/3) = (x - 5/2) * (y + 4/3) ∧ 
  (x + 5/2) * (y - 2/3) = x * y →
  x * y = 20 := by
  sorry

end rectangle_area_invariance_l1869_186947


namespace unique_solution_l1869_186973

theorem unique_solution : ∃! (x y : ℕ+), x^(y:ℕ) + 1 = y^(x:ℕ) ∧ 2*(x^(y:ℕ)) = y^(x:ℕ) + 13 :=
by
  sorry

end unique_solution_l1869_186973


namespace total_sum_lent_l1869_186935

/-- Represents the sum of money lent in two parts -/
structure LoanParts where
  first : ℝ
  second : ℝ

/-- Calculates the interest for a given principal, rate, and time -/
def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent (loan : LoanParts) :
  loan.second = 1648 →
  interest loan.first 0.03 8 = interest loan.second 0.05 3 →
  loan.first + loan.second = 2678 := by
  sorry

end total_sum_lent_l1869_186935


namespace positive_real_equalities_l1869_186972

theorem positive_real_equalities (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c) ∧
  ((a + b + c) * (a^2 + b^2 + c^2 - a*b - b*c - a*c) = 0 → a = b ∧ b = c) ∧
  (a^4 + b^4 + c^4 + d^4 = 4*a*b*c*d → a = b ∧ b = c ∧ c = d) := by
  sorry

end positive_real_equalities_l1869_186972


namespace units_digit_of_4659_to_157_l1869_186980

theorem units_digit_of_4659_to_157 :
  (4659^157) % 10 = 9 := by sorry

end units_digit_of_4659_to_157_l1869_186980


namespace handshakes_in_specific_convention_l1869_186986

/-- Represents a convention with companies and representatives -/
structure Convention where
  num_companies : ℕ
  reps_per_company : ℕ
  companies_to_shake : ℕ

/-- Calculates the total number of handshakes in the convention -/
def total_handshakes (conv : Convention) : ℕ :=
  let total_people := conv.num_companies * conv.reps_per_company
  let handshakes_per_person := (conv.companies_to_shake * conv.reps_per_company)
  (total_people * handshakes_per_person) / 2

/-- The specific convention described in the problem -/
def specific_convention : Convention :=
  { num_companies := 5
  , reps_per_company := 4
  , companies_to_shake := 2 }

theorem handshakes_in_specific_convention :
  total_handshakes specific_convention = 80 := by
  sorry


end handshakes_in_specific_convention_l1869_186986


namespace square_diff_cubed_l1869_186981

theorem square_diff_cubed : (7^2 - 5^2)^3 = 13824 := by
  sorry

end square_diff_cubed_l1869_186981


namespace function_range_l1869_186909

theorem function_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - a*x + a + 3 < 0 ∧ x - a < 0)) → 
  a ∈ Set.Icc (-3) 6 := by
  sorry

end function_range_l1869_186909


namespace max_people_satisfying_conditions_l1869_186979

/-- Represents a group of people and their relationships -/
structure PeopleGroup where
  n : ℕ
  knows : Fin n → Fin n → Prop
  knows_sym : ∀ i j, knows i j ↔ knows j i

/-- Any 3 people have at least 2 who know each other -/
def condition_a (g : PeopleGroup) : Prop :=
  ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    g.knows i j ∨ g.knows j k ∨ g.knows i k

/-- Any 4 people have at least 2 who don't know each other -/
def condition_b (g : PeopleGroup) : Prop :=
  ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ l →
    ¬g.knows i j ∨ ¬g.knows i k ∨ ¬g.knows i l ∨
    ¬g.knows j k ∨ ¬g.knows j l ∨ ¬g.knows k l

/-- The maximum number of people satisfying both conditions is 8 -/
theorem max_people_satisfying_conditions :
  (∃ g : PeopleGroup, g.n = 8 ∧ condition_a g ∧ condition_b g) ∧
  (∀ g : PeopleGroup, condition_a g ∧ condition_b g → g.n ≤ 8) :=
sorry

end max_people_satisfying_conditions_l1869_186979


namespace BC_vector_l1869_186977

def complex_vector (a b : ℂ) : ℂ := b - a

theorem BC_vector (OA OC AB : ℂ) 
  (h1 : OA = -2 + I) 
  (h2 : OC = 3 + 2*I) 
  (h3 : AB = 1 + 5*I) : 
  complex_vector (OA + AB) OC = 4 - 4*I := by
  sorry

end BC_vector_l1869_186977


namespace consecutive_integers_sum_l1869_186902

theorem consecutive_integers_sum (a b c d : ℝ) : 
  (b = a + 1 ∧ c = b + 1 ∧ d = c + 1) →  -- consecutive integers condition
  (a + d = 180) →                        -- sum of first and fourth is 180
  b = 90.5 :=                            -- second integer is 90.5
by sorry

end consecutive_integers_sum_l1869_186902


namespace equation_solutions_l1869_186938

theorem equation_solutions : 
  (∃ s1 : Set ℝ, s1 = {x : ℝ | x^2 + 2*x - 8 = 0} ∧ s1 = {-4, 2}) ∧ 
  (∃ s2 : Set ℝ, s2 = {x : ℝ | x*(x-2) = x-2} ∧ s2 = {2, 1}) := by sorry

end equation_solutions_l1869_186938


namespace number_divided_by_0_025_equals_40_l1869_186948

theorem number_divided_by_0_025_equals_40 (x : ℝ) : x / 0.025 = 40 → x = 1 := by
  sorry

end number_divided_by_0_025_equals_40_l1869_186948


namespace fred_dimes_problem_l1869_186985

/-- Represents the number of dimes Fred's sister borrowed -/
def dimes_borrowed (initial_dimes remaining_dimes : ℕ) : ℕ :=
  initial_dimes - remaining_dimes

theorem fred_dimes_problem (initial_dimes remaining_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : remaining_dimes = 4) :
  dimes_borrowed initial_dimes remaining_dimes = 3 := by
  sorry

end fred_dimes_problem_l1869_186985


namespace oxen_equivalence_l1869_186908

/-- The amount of fodder a buffalo eats per day -/
def buffalo_fodder : ℝ := sorry

/-- The amount of fodder a cow eats per day -/
def cow_fodder : ℝ := sorry

/-- The amount of fodder an ox eats per day -/
def ox_fodder : ℝ := sorry

/-- The total amount of fodder available -/
def total_fodder : ℝ := sorry

theorem oxen_equivalence :
  (3 * buffalo_fodder = 4 * cow_fodder) →
  (15 * buffalo_fodder + 8 * ox_fodder + 24 * cow_fodder) * 36 = total_fodder →
  (30 * buffalo_fodder + 8 * ox_fodder + 64 * cow_fodder) * 18 = total_fodder →
  (∃ n : ℕ, n * ox_fodder = 3 * buffalo_fodder ∧ n * ox_fodder = 4 * cow_fodder ∧ n = 4) :=
by sorry

end oxen_equivalence_l1869_186908


namespace zoo_arrangement_count_l1869_186924

def num_lions : Nat := 3
def num_zebras : Nat := 4
def num_monkeys : Nat := 6
def total_animals : Nat := num_lions + num_zebras + num_monkeys

theorem zoo_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_lions) * (Nat.factorial num_zebras) * (Nat.factorial num_monkeys) = 622080 :=
by sorry

end zoo_arrangement_count_l1869_186924


namespace local_minimum_at_negative_one_l1869_186942

open Real

/-- The function f(x) = xe^x has a local minimum at x = -1 -/
theorem local_minimum_at_negative_one (f : ℝ → ℝ) (h : ∀ x, f x = x * exp x) :
  IsLocalMin f (-1) :=
sorry

end local_minimum_at_negative_one_l1869_186942


namespace buttons_per_shirt_l1869_186941

/-- Given Jack's shirt-making scenario, prove the number of buttons per shirt. -/
theorem buttons_per_shirt (num_kids : ℕ) (shirts_per_kid : ℕ) (total_buttons : ℕ) : 
  num_kids = 3 →
  shirts_per_kid = 3 →
  total_buttons = 63 →
  ∃ (buttons_per_shirt : ℕ), 
    buttons_per_shirt * (num_kids * shirts_per_kid) = total_buttons ∧
    buttons_per_shirt = 7 :=
by sorry

end buttons_per_shirt_l1869_186941


namespace student_mistake_difference_l1869_186982

theorem student_mistake_difference (number : ℕ) (h : number = 192) : 
  (5 / 6 : ℚ) * number - (5 / 16 : ℚ) * number = 100 := by
  sorry

end student_mistake_difference_l1869_186982


namespace cos_negative_75_degrees_l1869_186968

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_negative_75_degrees_l1869_186968


namespace john_profit_l1869_186917

/-- Calculate the selling price given the cost price and profit percentage -/
def selling_price (cost : ℚ) (profit_percent : ℚ) : ℚ :=
  cost * (1 + profit_percent / 100)

/-- Calculate the overall profit given the cost and selling prices of two items -/
def overall_profit (cost1 cost2 sell1 sell2 : ℚ) : ℚ :=
  (sell1 + sell2) - (cost1 + cost2)

theorem john_profit :
  let grinder_cost : ℚ := 15000
  let mobile_cost : ℚ := 10000
  let grinder_loss_percent : ℚ := 4
  let mobile_profit_percent : ℚ := 10
  let grinder_sell := selling_price grinder_cost (-grinder_loss_percent)
  let mobile_sell := selling_price mobile_cost mobile_profit_percent
  overall_profit grinder_cost mobile_cost grinder_sell mobile_sell = 400 := by
  sorry

end john_profit_l1869_186917


namespace race_overtake_equation_l1869_186916

/-- The time it takes for John to overtake Steve in a race --/
def overtake_time (initial_distance : ℝ) (john_initial_speed : ℝ) (john_acceleration : ℝ) (steve_speed : ℝ) (final_distance : ℝ) : ℝ → Prop :=
  λ t => 0.5 * john_acceleration * t^2 + john_initial_speed * t - steve_speed * t - initial_distance - final_distance = 0

theorem race_overtake_equation :
  let initial_distance : ℝ := 15
  let john_initial_speed : ℝ := 3.5
  let john_acceleration : ℝ := 0.25
  let steve_speed : ℝ := 3.8
  let final_distance : ℝ := 2
  ∃ t : ℝ, overtake_time initial_distance john_initial_speed john_acceleration steve_speed final_distance t :=
by sorry

end race_overtake_equation_l1869_186916


namespace adjacent_i_probability_is_one_fifth_l1869_186927

/-- The probability of forming a 10-letter code with two adjacent i's -/
def adjacent_i_probability : ℚ :=
  let total_arrangements := Nat.factorial 10
  let favorable_arrangements := Nat.factorial 9 * Nat.factorial 2
  favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of forming a 10-letter code
    with two adjacent i's is 1/5 -/
theorem adjacent_i_probability_is_one_fifth :
  adjacent_i_probability = 1 / 5 := by
  sorry

end adjacent_i_probability_is_one_fifth_l1869_186927


namespace salary_change_percentage_l1869_186951

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.3)
  let final_salary := decreased_salary * (1 + 0.3)
  (initial_salary - final_salary) / initial_salary * 100 = 9 := by
sorry

end salary_change_percentage_l1869_186951


namespace parallel_lines_slope_l1869_186906

/-- Theorem: For three parallel lines with y-intercepts 2, 3, and 4, 
    if the sum of their x-intercepts is 36, then their slope is -1/4. -/
theorem parallel_lines_slope (m : ℝ) 
  (h1 : m * (-2/m) + m * (-3/m) + m * (-4/m) = 36) : m = -1/4 := by
  sorry

end parallel_lines_slope_l1869_186906


namespace quadratic_roots_mean_l1869_186929

theorem quadratic_roots_mean (b c : ℝ) (r₁ r₂ : ℝ) : 
  (r₁ + r₂) / 2 = 9 →
  (r₁ * r₂).sqrt = 21 →
  r₁ + r₂ = -b →
  r₁ * r₂ = c →
  b = -18 ∧ c = 441 :=
by sorry

end quadratic_roots_mean_l1869_186929


namespace difference_of_numbers_l1869_186922

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 221) : 
  |x - y| = 4 := by sorry

end difference_of_numbers_l1869_186922


namespace tangent_line_intersection_l1869_186991

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_line_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧ 
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → 
      f a x ≠ (f_derivative a x₁) * x) :=
sorry

end tangent_line_intersection_l1869_186991


namespace xyz_inequality_l1869_186915

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end xyz_inequality_l1869_186915


namespace range_of_a_l1869_186930

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 = 0}
def B : Set ℝ := {1, 2}

theorem range_of_a (a : ℝ) : 
  (A a ∪ B = B) ↔ a ∈ Set.Icc (-2 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end range_of_a_l1869_186930


namespace polynomial_division_remainder_l1869_186966

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^12 - x^6 + 1 = (x^2 - 1) * q + 1 := by
  sorry

end polynomial_division_remainder_l1869_186966


namespace trapezoid_division_l1869_186988

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  area : ℝ
  base_ratio : ℝ
  smaller_base : ℝ
  larger_base : ℝ
  height : ℝ
  area_eq : area = (smaller_base + larger_base) * height / 2
  base_ratio_eq : larger_base = base_ratio * smaller_base

/-- Represents the two smaller trapezoids formed by the median line -/
structure SmallerTrapezoids where
  top_area : ℝ
  bottom_area : ℝ

/-- The main theorem stating the areas of smaller trapezoids -/
theorem trapezoid_division (t : Trapezoid) 
  (h1 : t.area = 80)
  (h2 : t.base_ratio = 3) :
  ∃ (st : SmallerTrapezoids), st.top_area = 30 ∧ st.bottom_area = 50 := by
  sorry

end trapezoid_division_l1869_186988


namespace special_triangle_not_necessarily_right_l1869_186969

/-- A triangle with sides a, b, and c where a² = 5, b² = 12, and c² = 13 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a^2 = 5
  hb : b^2 = 12
  hc : c^2 = 13

/-- A right triangle is a triangle where one of its angles is 90 degrees -/
def IsRightTriangle (t : SpecialTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

/-- Theorem stating that it cannot be determined if a SpecialTriangle is a right triangle -/
theorem special_triangle_not_necessarily_right (t : SpecialTriangle) :
  ¬ (IsRightTriangle t) := by sorry

end special_triangle_not_necessarily_right_l1869_186969


namespace triangle_side_length_l1869_186974

/-- Given a triangle ABC with side lengths a, b, c and angle B, 
    prove that if b = √3, c = 3, and B = 30°, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = Real.sqrt 3 → c = 3 → B = π / 6 → a = 2 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l1869_186974


namespace point_in_fourth_quadrant_implies_a_squared_plus_one_l1869_186920

theorem point_in_fourth_quadrant_implies_a_squared_plus_one (a : ℤ) : 
  (3 * a - 9 > 0) → (2 * a - 10 < 0) → a^2 + 1 = 17 := by
  sorry

end point_in_fourth_quadrant_implies_a_squared_plus_one_l1869_186920


namespace eggs_remaining_l1869_186994

def dozen : ℕ := 12

def initial_eggs (num_dozens : ℕ) : ℕ := num_dozens * dozen

def remaining_after_half (total : ℕ) : ℕ := total / 2

def final_eggs (after_half : ℕ) (broken : ℕ) : ℕ := after_half - broken

theorem eggs_remaining (num_dozens : ℕ) (broken : ℕ) 
  (h1 : num_dozens = 6) 
  (h2 : broken = 15) : 
  final_eggs (remaining_after_half (initial_eggs num_dozens)) broken = 21 := by
  sorry

#check eggs_remaining

end eggs_remaining_l1869_186994


namespace line_l_properties_l1869_186983

/-- The line l is defined by the equation (a^2 + a + 1)x - y + 1 = 0, where a is a real number -/
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  (a^2 + a + 1) * x - y + 1 = 0

/-- The perpendicular line is defined by the equation x + y = 0 -/
def perp_line (x y : ℝ) : Prop :=
  x + y = 0

theorem line_l_properties :
  (∀ a : ℝ, line_l a 0 1) ∧ 
  (∀ x y : ℝ, line_l 0 x y → perp_line x y) :=
by sorry

end line_l_properties_l1869_186983


namespace circle_on_grid_regions_l1869_186950

/-- Represents a grid with uniform spacing -/
structure Grid :=
  (spacing : ℝ)

/-- Represents a circle on the grid -/
structure CircleOnGrid :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents a region formed by circle arcs and grid line segments -/
structure Region

/-- Calculates the number of regions formed by a circle on a grid -/
def count_regions (g : Grid) (c : CircleOnGrid) : ℕ :=
  sorry

/-- Calculates the areas of regions formed by a circle on a grid -/
def region_areas (g : Grid) (c : CircleOnGrid) : List ℝ :=
  sorry

/-- Main theorem: Number and areas of regions formed by a circle on a grid -/
theorem circle_on_grid_regions 
  (g : Grid) 
  (c : CircleOnGrid) 
  (h1 : g.spacing = 1) 
  (h2 : c.radius = 5) 
  (h3 : c.center = (0, 0)) :
  (count_regions g c = 56) ∧ 
  (region_areas g c ≈ [0.966, 0.761, 0.317, 0.547]) :=
by sorry

#check circle_on_grid_regions

end circle_on_grid_regions_l1869_186950


namespace subway_speed_increase_l1869_186926

/-- The speed equation for the subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- The theorem stating the time at which the train is moving 55 km/h faster -/
theorem subway_speed_increase (s : ℝ) (h1 : 0 ≤ s) (h2 : s ≤ 7) :
  speed s = speed 2 + 55 ↔ s = 7 := by sorry

end subway_speed_increase_l1869_186926


namespace janes_breakfast_problem_l1869_186967

/-- Represents the number of breakfast items bought -/
structure BreakfastItems where
  muffins : ℕ
  bagels : ℕ
  croissants : ℕ

/-- Calculates the total cost in cents -/
def totalCost (items : BreakfastItems) : ℕ :=
  50 * items.muffins + 75 * items.bagels + 65 * items.croissants

theorem janes_breakfast_problem :
  ∃ (items : BreakfastItems),
    items.muffins + items.bagels + items.croissants = 6 ∧
    items.bagels = 2 ∧
    (totalCost items) % 100 = 0 ∧
    items.muffins = 4 :=
  sorry

end janes_breakfast_problem_l1869_186967


namespace largest_interesting_number_l1869_186996

/-- A real number is interesting if removing one digit from its decimal representation results in 2x -/
def IsInteresting (x : ℝ) : Prop :=
  ∃ (y : ℕ) (z : ℝ), 0 < x ∧ x < 1 ∧ x = y / 10 + z ∧ 2 * x = z

/-- The largest interesting number is 0.375 -/
theorem largest_interesting_number :
  IsInteresting (3 / 8) ∧ ∀ x : ℝ, IsInteresting x → x ≤ 3 / 8 :=
sorry

end largest_interesting_number_l1869_186996


namespace cone_height_l1869_186934

/-- Proves that a cone with lateral area 15π cm² and base radius 3 cm has a height of 4 cm -/
theorem cone_height (lateral_area : ℝ) (base_radius : ℝ) (height : ℝ) : 
  lateral_area = 15 * Real.pi ∧ base_radius = 3 → height = 4 := by
  sorry

#check cone_height

end cone_height_l1869_186934


namespace hyperbola_equation_l1869_186936

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

theorem hyperbola_equation :
  ∀ (x y : ℝ), 
    (∃ (t : ℝ), hyperbola t y ∧ asymptotes t y) →  -- Hyperbola exists with given asymptotes
    hyperbola 4 (Real.sqrt 3) →                    -- Hyperbola passes through (4, √3)
    hyperbola x y                                  -- The equation of the hyperbola
  := by sorry

end hyperbola_equation_l1869_186936


namespace arithmetic_average_characterization_l1869_186943

/-- φ(n) is the number of positive integers ≤ n and coprime with n -/
def phi (n : ℕ+) : ℕ := sorry

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- One of n, φ(n), or τ(n) is the arithmetic average of the other two -/
def is_arithmetic_average (n : ℕ+) : Prop :=
  (n : ℕ) = (phi n + tau n) / 2 ∨
  phi n = ((n : ℕ) + tau n) / 2 ∨
  tau n = ((n : ℕ) + phi n) / 2

theorem arithmetic_average_characterization (n : ℕ+) :
  is_arithmetic_average n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end arithmetic_average_characterization_l1869_186943


namespace john_relatives_money_l1869_186912

theorem john_relatives_money (grandpa : ℕ) : 
  grandpa = 30 → 
  (grandpa + 3 * grandpa + 2 * grandpa + (3 * grandpa) / 2 : ℕ) = 225 := by
  sorry

end john_relatives_money_l1869_186912


namespace equation_roots_theorem_l1869_186953

/-- 
Given an equation (x² - px) / (kx - d) = (n - 2) / (n + 2),
where the roots are numerically equal but opposite in sign and their product is 1,
prove that n = 2(k - p) / (k + p).
-/
theorem equation_roots_theorem (p k d n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - p*x) / (k*x - d) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  (∃ r : ℝ, x r * x (-r) = 1) →
  n = 2*(k - p) / (k + p) := by
  sorry

end equation_roots_theorem_l1869_186953


namespace fifth_graders_count_l1869_186970

def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def teachers_per_grade : ℕ := 4
def parents_per_grade : ℕ := 2
def num_buses : ℕ := 5
def seats_per_bus : ℕ := 72

def total_seats : ℕ := num_buses * seats_per_bus
def total_chaperones : ℕ := (teachers_per_grade + parents_per_grade) * 3
def sixth_and_seventh : ℕ := sixth_graders + seventh_graders
def seats_taken : ℕ := sixth_and_seventh + total_chaperones

theorem fifth_graders_count : 
  total_seats - seats_taken = 109 := by sorry

end fifth_graders_count_l1869_186970


namespace regular_polygon_exterior_angle_l1869_186933

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) :
  n > 2 ∧ exterior_angle = 72 → n = 5 := by
  sorry

end regular_polygon_exterior_angle_l1869_186933
