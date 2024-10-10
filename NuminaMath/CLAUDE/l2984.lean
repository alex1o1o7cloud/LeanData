import Mathlib

namespace factorization_x4_plus_81_l2984_298437

theorem factorization_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end factorization_x4_plus_81_l2984_298437


namespace backpack_cost_relationship_l2984_298406

theorem backpack_cost_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Cost of type A backpacks is positive
  (h3 : 600 > 0) -- Cost of type B backpacks is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 
  810 / (x + 20) = (600 / x) * (1 - 0.1) :=
sorry

end backpack_cost_relationship_l2984_298406


namespace main_theorem_l2984_298494

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def even_symmetric (f : ℝ → ℝ) : Prop := ∀ x, f x - f (-x) = 0

def symmetric_about_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (2 - x)

def increasing_on_zero_two (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorems to be proved
def periodic_four (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

def decreasing_on_two_four (f : ℝ → ℝ) : Prop := 
  ∀ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f y < f x

-- Main theorem
theorem main_theorem (heven : even_symmetric f) 
                     (hsym : symmetric_about_two f) 
                     (hinc : increasing_on_zero_two f) : 
  periodic_four f ∧ decreasing_on_two_four f := by
  sorry

end main_theorem_l2984_298494


namespace car_instantaneous_speed_l2984_298433

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - t^2 + 2

-- State the theorem
theorem car_instantaneous_speed : 
  (deriv s) 1 = 4 := by sorry

end car_instantaneous_speed_l2984_298433


namespace point_inside_iff_odd_intersections_l2984_298438

/-- A closed, non-self-intersecting path in a plane. -/
structure ClosedPath :=
  (path : Set (ℝ × ℝ))
  (closed : IsClosed path)
  (non_self_intersecting : ∀ x y : ℝ × ℝ, x ∈ path → y ∈ path → x ≠ y → (∃ t : ℝ, 0 < t ∧ t < 1 ∧ (1 - t) • x + t • y ∉ path))

/-- A point in the plane. -/
def Point := ℝ × ℝ

/-- The number of intersections between a line segment and a path. -/
def intersectionCount (p q : Point) (path : ClosedPath) : ℕ :=
  sorry

/-- A point is known to be outside the region bounded by the path. -/
def isOutside (p : Point) (path : ClosedPath) : Prop :=
  sorry

/-- A point is inside the region bounded by the path. -/
def isInside (p : Point) (path : ClosedPath) : Prop :=
  ∀ q : Point, isOutside q path → Odd (intersectionCount p q path)

theorem point_inside_iff_odd_intersections (p : Point) (path : ClosedPath) :
  isInside p path ↔ ∀ q : Point, isOutside q path → Odd (intersectionCount p q path) :=
sorry

end point_inside_iff_odd_intersections_l2984_298438


namespace banana_sharing_l2984_298450

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 :=
by
  sorry

end banana_sharing_l2984_298450


namespace passenger_gate_probability_l2984_298459

def num_gates : ℕ := 15
def distance_between_gates : ℕ := 90
def max_walking_distance : ℕ := 360

theorem passenger_gate_probability : 
  let total_possibilities := num_gates * (num_gates - 1)
  let valid_possibilities := (
    2 * (4 + 5 + 6 + 7) +  -- Gates 1,2,3,4 and 12,13,14,15
    4 * 8 +                -- Gates 5,6,10,11
    3 * 8                  -- Gates 7,8,9
  )
  (valid_possibilities : ℚ) / total_possibilities = 10 / 21 :=
sorry

end passenger_gate_probability_l2984_298459


namespace polygon_sides_l2984_298471

theorem polygon_sides (interior_angle_sum : ℝ) : interior_angle_sum = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = interior_angle_sum := by
  sorry

end polygon_sides_l2984_298471


namespace baker_cakes_problem_l2984_298488

theorem baker_cakes_problem (sold : ℕ) (left : ℕ) (h1 : sold = 41) (h2 : left = 13) :
  sold + left = 54 :=
by sorry

end baker_cakes_problem_l2984_298488


namespace inequality_proof_l2984_298447

theorem inequality_proof (a : ℝ) (ha : a > 0) : 2 * a / (1 + a^2) ≤ 1 := by
  sorry

end inequality_proof_l2984_298447


namespace domain_shift_l2984_298499

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_shift (h : ∀ x, f (x + 1) ∈ domain_f_shifted ↔ x ∈ domain_f_shifted) :
  (∀ x, f x ∈ Set.Icc (-1) 4 ↔ x ∈ Set.Icc (-1) 4) :=
sorry

end domain_shift_l2984_298499


namespace largest_mu_inequality_l2984_298462

theorem largest_mu_inequality : 
  ∃ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 3*c*d) ∧ 
  (∀ (μ' : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 3*c*d) → μ' ≤ μ) ∧ 
  μ = 1 := by
  sorry

end largest_mu_inequality_l2984_298462


namespace area_fold_points_specific_triangle_l2984_298407

/-- Represents a right triangle ABC -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  angleB : ℝ

/-- Represents the area of fold points -/
def area_fold_points (t : RightTriangle) : ℝ := sorry

/-- Main theorem: Area of fold points for the given right triangle -/
theorem area_fold_points_specific_triangle :
  let t : RightTriangle := { AB := 45, AC := 90, angleB := 90 }
  area_fold_points t = 379 * Real.pi :=
by sorry

end area_fold_points_specific_triangle_l2984_298407


namespace annulus_equal_area_division_l2984_298441

theorem annulus_equal_area_division (r : ℝ) : 
  r > 0 ∧ r < 14 ∧ 
  (π * (14^2 - r^2) = π * (r^2 - 2^2)) → 
  r = 10 := by sorry

end annulus_equal_area_division_l2984_298441


namespace fred_stickers_l2984_298453

theorem fred_stickers (jerry george fred : ℕ) 
  (h1 : jerry = 3 * george)
  (h2 : george = fred - 6)
  (h3 : jerry = 36) : 
  fred = 18 := by
sorry

end fred_stickers_l2984_298453


namespace inequality_system_solution_set_l2984_298472

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by sorry

end inequality_system_solution_set_l2984_298472


namespace equation_has_real_roots_l2984_298412

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end equation_has_real_roots_l2984_298412


namespace commission_calculation_l2984_298481

/-- The commission calculation problem -/
theorem commission_calculation
  (commission_rate : ℝ)
  (total_sales : ℝ)
  (h1 : commission_rate = 0.04)
  (h2 : total_sales = 312.5) :
  commission_rate * total_sales = 12.5 := by
  sorry

end commission_calculation_l2984_298481


namespace partial_fraction_decomposition_l2984_298435

theorem partial_fraction_decomposition (x : ℝ) (h : x ≠ 0) :
  (-2 * x^2 + 5 * x - 6) / (x^3 + 2 * x) = 
  (-3 : ℝ) / x + (x + 5) / (x^2 + 2) :=
by sorry

end partial_fraction_decomposition_l2984_298435


namespace slope_angle_of_line_l2984_298454

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y + 5 = 0 →
  Real.arctan (-Real.sqrt 3 / 3) = 150 * π / 180 :=
by sorry

end slope_angle_of_line_l2984_298454


namespace bill_apples_left_l2984_298496

/-- The number of apples Bill has left after distributing to teachers and baking pies -/
def apples_left (initial_apples : ℕ) (num_children : ℕ) (apples_per_teacher : ℕ) 
  (num_teachers_per_child : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_children * apples_per_teacher * num_teachers_per_child) - (num_pies * apples_per_pie)

/-- Theorem stating that Bill has 18 apples left -/
theorem bill_apples_left : 
  apples_left 50 2 3 2 2 10 = 18 := by
  sorry

end bill_apples_left_l2984_298496


namespace x_value_l2984_298463

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by sorry

end x_value_l2984_298463


namespace soap_weight_calculation_l2984_298452

/-- Calculates the total weight of soap bars given the weights of other items and suitcase weights. -/
theorem soap_weight_calculation (initial_weight final_weight perfume_weight chocolate_weight jam_weight : ℝ) : 
  initial_weight = 5 →
  perfume_weight = 5 * 1.2 / 16 →
  chocolate_weight = 4 →
  jam_weight = 2 * 8 / 16 →
  final_weight = 11 →
  final_weight - initial_weight - (perfume_weight + chocolate_weight + jam_weight) = 0.625 := by
  sorry

#check soap_weight_calculation

end soap_weight_calculation_l2984_298452


namespace daily_income_ratio_l2984_298448

/-- The ratio of daily income in a business where:
    - Initial income on day 1 is 3
    - Income on day 15 is 36
    - Each day's income is a multiple of the previous day's income
-/
theorem daily_income_ratio : ∃ (r : ℝ), 
  r > 0 ∧ 
  3 * r^14 = 36 ∧ 
  r = 2^(1/7) * 3^(1/14) := by
  sorry

end daily_income_ratio_l2984_298448


namespace inequality_system_solution_set_l2984_298478

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 5 < 4 ∧ (3 * x + 1) / 2 ≥ 2 * x - 1}
  S = {x : ℝ | x < -1} := by
  sorry

end inequality_system_solution_set_l2984_298478


namespace kittens_given_to_friends_l2984_298442

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa has left -/
def remaining_kittens : ℕ := 4

/-- The number of kittens Alyssa gave to her friends -/
def kittens_given_away : ℕ := initial_kittens - remaining_kittens

theorem kittens_given_to_friends :
  kittens_given_away = 4 := by sorry

end kittens_given_to_friends_l2984_298442


namespace jerome_bicycle_trip_distance_l2984_298423

/-- The total distance of Jerome's bicycle trip -/
def total_distance (daily_distance : ℕ) (num_days : ℕ) (last_day_distance : ℕ) : ℕ :=
  daily_distance * num_days + last_day_distance

/-- Theorem stating that Jerome's bicycle trip is 150 miles long -/
theorem jerome_bicycle_trip_distance :
  total_distance 12 12 6 = 150 := by
  sorry

end jerome_bicycle_trip_distance_l2984_298423


namespace crayon_production_l2984_298491

theorem crayon_production (num_colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) :
  num_colors = 4 →
  crayons_per_color = 2 →
  boxes_per_hour = 5 →
  hours = 4 →
  (num_colors * crayons_per_color * boxes_per_hour * hours) = 160 :=
by sorry

end crayon_production_l2984_298491


namespace x_percent_of_x_squared_l2984_298401

theorem x_percent_of_x_squared (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 16) : x = 12 := by
  sorry

end x_percent_of_x_squared_l2984_298401


namespace ice_cream_survey_l2984_298446

theorem ice_cream_survey (total : ℕ) (strawberry_percent : ℚ) (vanilla_percent : ℚ) (chocolate_percent : ℚ)
  (h_total : total = 500)
  (h_strawberry : strawberry_percent = 46 / 100)
  (h_vanilla : vanilla_percent = 71 / 100)
  (h_chocolate : chocolate_percent = 85 / 100) :
  ∃ (all_three : ℕ), all_three ≥ 10 ∧
    (strawberry_percent * total + vanilla_percent * total + chocolate_percent * total
      = (total - all_three) * 2 + all_three * 3) :=
by sorry

end ice_cream_survey_l2984_298446


namespace min_sum_of_product_3960_l2984_298487

theorem min_sum_of_product_3960 (a b c : ℕ+) (h : a * b * c = 3960) :
  (∀ x y z : ℕ+, x * y * z = 3960 → a + b + c ≤ x + y + z) ∧ a + b + c = 72 := by
  sorry

end min_sum_of_product_3960_l2984_298487


namespace allocation_schemes_eq_1080_l2984_298417

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to divide 6 families into 4 groups and allocate to 4 villages --/
def allocation_schemes : ℕ :=
  let group_formations := choose 6 2 * choose 4 2 * choose 2 1 * choose 1 1 / (arrange 2 2 * arrange 2 2)
  let village_allocations := arrange 4 4
  group_formations * village_allocations

theorem allocation_schemes_eq_1080 : allocation_schemes = 1080 := by
  sorry

end allocation_schemes_eq_1080_l2984_298417


namespace weekly_pay_solution_l2984_298483

def weekly_pay_problem (y_pay : ℝ) (x_percentage : ℝ) : Prop :=
  let x_pay := x_percentage * y_pay
  x_pay + y_pay = 638

theorem weekly_pay_solution :
  weekly_pay_problem 290 1.2 :=
by sorry

end weekly_pay_solution_l2984_298483


namespace both_in_picture_probability_l2984_298419

/-- Alice's lap time in seconds -/
def alice_lap_time : ℕ := 120

/-- Bob's lap time in seconds -/
def bob_lap_time : ℕ := 75

/-- Bob's start delay in seconds -/
def bob_start_delay : ℕ := 15

/-- Duration of one-third of the track for Alice in seconds -/
def alice_third_track : ℕ := alice_lap_time / 3

/-- Duration of one-third of the track for Bob in seconds -/
def bob_third_track : ℕ := bob_lap_time / 3

/-- Least common multiple of Alice and Bob's lap times -/
def lcm_lap_times : ℕ := lcm alice_lap_time bob_lap_time

/-- Time window for taking the picture in seconds -/
def picture_window : ℕ := 60

/-- Probability of both Alice and Bob being in the picture -/
def probability_both_in_picture : ℚ := 11 / 1200

theorem both_in_picture_probability :
  probability_both_in_picture = 11 / 1200 := by
  sorry

end both_in_picture_probability_l2984_298419


namespace rotation_90_degrees_l2984_298405

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (8 - 5 * Complex.I) = 5 + 8 * Complex.I := by sorry

end rotation_90_degrees_l2984_298405


namespace negative_eight_to_negative_four_thirds_l2984_298464

theorem negative_eight_to_negative_four_thirds :
  Real.rpow (-8 : ℝ) (-4/3 : ℝ) = (1/16 : ℝ) := by
  sorry

end negative_eight_to_negative_four_thirds_l2984_298464


namespace equation_root_l2984_298492

theorem equation_root : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end equation_root_l2984_298492


namespace shaded_area_l2984_298414

/-- Given a square and a rhombus with shared side, calculates the area of the region inside the square but outside the rhombus -/
theorem shaded_area (square_area rhombus_area : ℝ) : 
  square_area = 25 →
  rhombus_area = 20 →
  ∃ (shaded_area : ℝ), shaded_area = square_area - (rhombus_area * 0.7) ∧ shaded_area = 11 := by
  sorry

end shaded_area_l2984_298414


namespace good_games_count_l2984_298439

def games_from_friend : ℕ := 50
def games_from_garage_sale : ℕ := 27
def non_working_games : ℕ := 74

def total_games : ℕ := games_from_friend + games_from_garage_sale

theorem good_games_count : total_games - non_working_games = 3 := by
  sorry

end good_games_count_l2984_298439


namespace inequality_not_always_true_l2984_298409

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, (a - b) / c > 0) :=
sorry

end inequality_not_always_true_l2984_298409


namespace cube_has_twelve_edges_l2984_298477

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where
  -- We don't need to specify any fields for this definition

/-- The number of edges in a cube. -/
def num_edges (c : Cube) : ℕ := 12

/-- Theorem: A cube has 12 edges. -/
theorem cube_has_twelve_edges (c : Cube) : num_edges c = 12 := by
  sorry

end cube_has_twelve_edges_l2984_298477


namespace expression_value_l2984_298434

theorem expression_value (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / a = 10 / 27 := by sorry

end expression_value_l2984_298434


namespace square_difference_division_problem_solution_l2984_298467

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
sorry

theorem problem_solution : (112^2 - 97^2) / 15 = 209 := by
  have h : 112 > 97 := by sorry
  have key := square_difference_division 112 97 h
  sorry

end square_difference_division_problem_solution_l2984_298467


namespace tangent_line_equation_l2984_298470

/-- A line passing through (2,0) and tangent to y = 1/x has equation x + y - 2 = 0 -/
theorem tangent_line_equation : ∃ (k : ℝ),
  (∀ x y : ℝ, y = k * (x - 2) → y = 1 / x → x * x * k - 2 * x * k - 1 = 0) ∧
  (4 * k * k + 4 * k = 0) ∧
  (∀ x y : ℝ, y = k * (x - 2) ↔ x + y - 2 = 0) :=
by sorry


end tangent_line_equation_l2984_298470


namespace cosine_squared_sum_equality_l2984_298443

theorem cosine_squared_sum_equality (x : ℝ) : 
  (Real.cos x)^2 + (Real.cos (2 * x))^2 + (Real.cos (3 * x))^2 = 1 ↔ 
  (∃ k : ℤ, x = (k * Real.pi / 2 + Real.pi / 4) ∨ x = (k * Real.pi / 3 + Real.pi / 6)) :=
by sorry

end cosine_squared_sum_equality_l2984_298443


namespace intersection_point_x_coordinate_l2984_298421

theorem intersection_point_x_coordinate 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.log x) 
  (P Q : ℝ × ℝ) 
  (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (hPQ : P.1 < Q.1) 
  (R : ℝ × ℝ) 
  (hR : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) 
  (T : ℝ × ℝ) 
  (hT : T.2 = R.2 ∧ f T.1 = T.2 ∧ T.1 ≠ R.1) : 
  T.1 = Real.sqrt 1000 := by
sorry

end intersection_point_x_coordinate_l2984_298421


namespace max_ate_second_most_l2984_298416

-- Define the children as a finite type
inductive Child : Type
  | Chris : Child
  | Max : Child
  | Brandon : Child
  | Kayla : Child
  | Tanya : Child

-- Define the eating relation
def ate_more_than (a b : Child) : Prop := sorry

-- Define the conditions
axiom chris_ate_more_than_max : ate_more_than Child.Chris Child.Max
axiom brandon_ate_less_than_kayla : ate_more_than Child.Kayla Child.Brandon
axiom kayla_ate_less_than_max : ate_more_than Child.Max Child.Kayla
axiom kayla_ate_more_than_tanya : ate_more_than Child.Kayla Child.Tanya

-- Define what it means to be the second most
def is_second_most (c : Child) : Prop :=
  ∃ (first : Child), (first ≠ c) ∧
    (∀ (other : Child), other ≠ first → other ≠ c → ate_more_than c other)

-- The theorem to prove
theorem max_ate_second_most : is_second_most Child.Max := by sorry

end max_ate_second_most_l2984_298416


namespace danny_bottle_caps_l2984_298461

theorem danny_bottle_caps (found_new : ℕ) (total_after : ℕ) (difference : ℕ) 
  (h1 : found_new = 50)
  (h2 : total_after = 60)
  (h3 : found_new = difference + 44) : 
  found_new - difference = 6 := by
sorry

end danny_bottle_caps_l2984_298461


namespace triangle_properties_l2984_298430

-- Define an acute triangle ABC
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  acute : angle_A > 0 ∧ angle_A < Real.pi/2 ∧
          angle_B > 0 ∧ angle_B < Real.pi/2 ∧
          angle_C > 0 ∧ angle_C < Real.pi/2

-- State the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = t.a * t.b)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.angle_C = (3 * Real.sqrt 3) / 2) :
  t.angle_C = Real.pi/3 ∧ t.a + t.b = 5 := by
  sorry

end triangle_properties_l2984_298430


namespace binomial_coefficient_1000_l2984_298466

theorem binomial_coefficient_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end binomial_coefficient_1000_l2984_298466


namespace minimum_guests_l2984_298431

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ 
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end minimum_guests_l2984_298431


namespace largest_n_for_equation_l2984_298480

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) ∧
    (∀ (m : ℕ), m > n → 
      ¬(∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
        m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12))) ∧
  (∀ (n : ℕ), (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) →
    n ≤ 6) := by
  sorry

end largest_n_for_equation_l2984_298480


namespace garden_tulips_calculation_l2984_298400

-- Define the initial state of the garden
def initial_daisies : ℕ := 32
def ratio_tulips : ℕ := 3
def ratio_daisies : ℕ := 4
def added_daisies : ℕ := 8

-- Define the function to calculate tulips based on daisies and ratio
def calculate_tulips (daisies : ℕ) : ℕ :=
  (daisies * ratio_tulips) / ratio_daisies

-- Theorem statement
theorem garden_tulips_calculation :
  let initial_tulips := calculate_tulips initial_daisies
  let final_daisies := initial_daisies + added_daisies
  let final_tulips := calculate_tulips final_daisies
  let additional_tulips := final_tulips - initial_tulips
  (additional_tulips = 6) ∧ (final_tulips = 30) := by
  sorry

end garden_tulips_calculation_l2984_298400


namespace infinite_geometric_series_sum_l2984_298457

theorem infinite_geometric_series_sum : 
  let a : ℚ := 2/5
  let r : ℚ := 1/2
  let series_sum := a / (1 - r)
  series_sum = 4/5 := by sorry

end infinite_geometric_series_sum_l2984_298457


namespace problem_solution_l2984_298410

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + 2 * Real.sqrt (a + b + c - d)) : 
  d = 1/2 := by
  sorry

end problem_solution_l2984_298410


namespace min_side_length_is_optimal_l2984_298425

/-- The minimum side length of a square piece of land with an area of at least 400 square feet -/
def min_side_length : ℝ := 20

/-- The area of the square land is at least 400 square feet -/
axiom area_constraint : min_side_length ^ 2 ≥ 400

/-- The minimum side length is optimal -/
theorem min_side_length_is_optimal :
  ∀ s : ℝ, s ^ 2 ≥ 400 → s ≥ min_side_length :=
by sorry

end min_side_length_is_optimal_l2984_298425


namespace find_m_value_l2984_298455

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end find_m_value_l2984_298455


namespace four_steps_on_number_line_l2984_298404

/-- Given a number line with equally spaced markings where the distance from 0 to 25
    is covered in 7 steps, prove that the number reached after 4 steps from 0 is 100/7. -/
theorem four_steps_on_number_line :
  ∀ (step_length : ℚ),
  step_length * 7 = 25 →
  4 * step_length = 100 / 7 := by
sorry

end four_steps_on_number_line_l2984_298404


namespace faster_bike_speed_l2984_298479

/-- Proves that given two motorbikes traveling the same distance,
    where one bike is faster and takes 1 hour less than the other bike,
    the speed of the faster bike is 60 kmph. -/
theorem faster_bike_speed
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_diff : ℝ)
  (h1 : distance = 960)
  (h2 : speed_fast = 60)
  (h3 : time_diff = 1)
  (h4 : distance / speed_fast + time_diff = distance / (distance / (distance / speed_fast + time_diff))) :
  speed_fast = 60 := by
  sorry

end faster_bike_speed_l2984_298479


namespace no_positive_sequence_with_recurrence_l2984_298485

theorem no_positive_sequence_with_recurrence : 
  ¬ ∃ (a : ℕ → ℝ), 
    (∀ n, a n > 0) ∧ 
    (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) := by
  sorry

end no_positive_sequence_with_recurrence_l2984_298485


namespace inequality_solutions_l2984_298489

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def solution_set2 : Set ℝ := {x : ℝ | x < -2 ∨ x > 3}

-- State the theorem
theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x - x^2 + 6 < 0 ↔ x ∈ solution_set2) := by
sorry

end inequality_solutions_l2984_298489


namespace angle_inequality_l2984_298426

open Real

theorem angle_inequality (a b c : ℝ) 
  (ha : a = sin (33 * π / 180))
  (hb : b = cos (55 * π / 180))
  (hc : c = tan (55 * π / 180)) :
  c > b ∧ b > a :=
by sorry

end angle_inequality_l2984_298426


namespace floor_of_3_2_l2984_298424

theorem floor_of_3_2 : ⌊(3.2 : ℝ)⌋ = 3 := by sorry

end floor_of_3_2_l2984_298424


namespace spherical_coordinate_negation_l2984_298445

/-- Given a point with rectangular coordinates (-3, 5, -2) and corresponding
    spherical coordinates (r, θ, φ), prove that the point with spherical
    coordinates (r, -θ, φ) has rectangular coordinates (-3, -5, -2). -/
theorem spherical_coordinate_negation (r θ φ : ℝ) :
  (r * Real.sin φ * Real.cos θ = -3 ∧
   r * Real.sin φ * Real.sin θ = 5 ∧
   r * Real.cos φ = -2) →
  (r * Real.sin φ * Real.cos (-θ) = -3 ∧
   r * Real.sin φ * Real.sin (-θ) = -5 ∧
   r * Real.cos φ = -2) := by
  sorry


end spherical_coordinate_negation_l2984_298445


namespace negation_equivalence_l2984_298408

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end negation_equivalence_l2984_298408


namespace neg_p_sufficient_not_necessary_for_neg_q_l2984_298476

theorem neg_p_sufficient_not_necessary_for_neg_q :
  let p := {x : ℝ | x < -1}
  let q := {x : ℝ | x < -4}
  (∀ x, x ∉ p → x ∉ q) ∧ (∃ x, x ∉ q ∧ x ∈ p) := by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l2984_298476


namespace book_arrangement_l2984_298436

theorem book_arrangement (n : ℕ) (a b c : ℕ) (h1 : n = a + b + c) (h2 : a = 3) (h3 : b = 2) (h4 : c = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 420 := by
  sorry

end book_arrangement_l2984_298436


namespace unique_prime_pair_sum_73_l2984_298498

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_prime_pair_sum_73 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 73 :=
by sorry

end unique_prime_pair_sum_73_l2984_298498


namespace projectile_max_height_l2984_298456

/-- Represents the elevation of a projectile at time t -/
def elevation (t : ℝ) : ℝ := 200 * t - 10 * t^2

/-- The time at which the projectile reaches its maximum height -/
def max_height_time : ℝ := 10

theorem projectile_max_height :
  ∀ t : ℝ, elevation t ≤ elevation max_height_time ∧
  elevation max_height_time = 1000 := by
  sorry

#check projectile_max_height

end projectile_max_height_l2984_298456


namespace factor_expression_l2984_298482

theorem factor_expression (x : ℝ) : 
  (20 * x^4 + 100 * x^2 - 10) - (-5 * x^4 + 15 * x^2 - 10) = 5 * x^2 * (5 * x^2 + 17) := by
  sorry

end factor_expression_l2984_298482


namespace rug_overlap_problem_l2984_298474

theorem rug_overlap_problem (total_rug_area single_coverage double_coverage triple_coverage : ℝ) :
  total_rug_area = 200 →
  single_coverage + double_coverage + triple_coverage = 140 →
  double_coverage = 24 →
  triple_coverage = 18 :=
by sorry

end rug_overlap_problem_l2984_298474


namespace subset_condition_intersection_condition_l2984_298440

def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4/3 ≤ a ∧ a ≤ 2 := by sorry

theorem intersection_condition (a : ℝ) : A ∩ B a = {x | 3 < x ∧ x < 4} ↔ a = 3 := by sorry

end subset_condition_intersection_condition_l2984_298440


namespace sequence_bound_l2984_298490

theorem sequence_bound (x : ℕ → ℝ) (b : ℝ) : 
  (∀ n : ℕ, x (n + 1) = x n ^ 2 - 4 * x n) →
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃ k : ℕ, x k ≥ b) →
  b = (3 + Real.sqrt 21) / 2 :=
by sorry

end sequence_bound_l2984_298490


namespace xy_squared_value_l2984_298484

theorem xy_squared_value (x y : ℤ) (h : y^2 + 2*x^2*y^2 = 20*x^2 + 412) : 2*x*y^2 = 288 := by
  sorry

end xy_squared_value_l2984_298484


namespace f_max_min_implies_a_range_l2984_298402

/-- The function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: If f(x) = x^2 - 2x + 3 has a maximum of 3 and a minimum of 2 on [0, a], then a ∈ [1, 2] -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 := by
  sorry

#check f_max_min_implies_a_range

end f_max_min_implies_a_range_l2984_298402


namespace division_equality_l2984_298469

theorem division_equality : (203515 : ℕ) / 2015 = 101 := by
  sorry

end division_equality_l2984_298469


namespace polynomial_factorization_l2984_298451

theorem polynomial_factorization (x : ℝ) :
  x^15 + x^10 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1 := by
  sorry

end polynomial_factorization_l2984_298451


namespace cosine_roots_l2984_298473

theorem cosine_roots (t : ℝ) : 
  (32 * (Real.cos (6 * π / 180))^5 - 40 * (Real.cos (6 * π / 180))^3 + 10 * Real.cos (6 * π / 180) - Real.sqrt 3 = 0) →
  (32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3 = 0 ↔ 
    t = Real.cos (66 * π / 180) ∨ 
    t = Real.cos (78 * π / 180) ∨ 
    t = Real.cos (138 * π / 180) ∨ 
    t = Real.cos (150 * π / 180) ∨ 
    t = Real.cos (6 * π / 180)) :=
by sorry

end cosine_roots_l2984_298473


namespace complement_of_union_l2984_298444

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 9 ∧ x > 0}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {7, 8, 9} := by sorry

end complement_of_union_l2984_298444


namespace demokhar_lifespan_l2984_298427

theorem demokhar_lifespan :
  ∀ (x : ℚ),
  (1 / 4 : ℚ) * x + (1 / 5 : ℚ) * x + (1 / 3 : ℚ) * x + 13 = x →
  x = 60 := by
sorry

end demokhar_lifespan_l2984_298427


namespace arithmetic_subsequence_multiples_of_3_l2984_298418

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The subsequence of an arithmetic sequence with indices that are multiples of 3 -/
def SubsequenceMultiplesOf3 (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a (3 * n)

theorem arithmetic_subsequence_multiples_of_3 (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d → SubsequenceMultiplesOf3 a b →
  ArithmeticSequence b (3 * d) :=
by sorry

end arithmetic_subsequence_multiples_of_3_l2984_298418


namespace potion_combinations_eq_thirteen_l2984_298497

/-- The number of ways to combine roots and minerals for a potion. -/
def potionCombinations : ℕ :=
  let totalRoots : ℕ := 3
  let totalMinerals : ℕ := 5
  let incompatibleCombinations : ℕ := 2
  totalRoots * totalMinerals - incompatibleCombinations

/-- Theorem stating that the number of potion combinations is 13. -/
theorem potion_combinations_eq_thirteen : potionCombinations = 13 := by
  sorry

end potion_combinations_eq_thirteen_l2984_298497


namespace smallest_n_value_l2984_298422

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ m : ℕ, m > 70 ∧ 70 ∣ (21 * m) → m ≥ N) → N = 80 :=
by sorry

end smallest_n_value_l2984_298422


namespace triangle_angle_ratio_l2984_298403

theorem triangle_angle_ratio (right_angle top_angle left_angle : ℝ) : 
  right_angle = 60 →
  top_angle = 70 →
  left_angle + right_angle + top_angle = 180 →
  left_angle / right_angle = 5 / 6 :=
by
  sorry

end triangle_angle_ratio_l2984_298403


namespace infinite_common_elements_l2984_298465

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

def subsequence_a (k : ℕ) : ℤ := sequence_a (2 * k + 1)

def subsequence_b (k : ℕ) : ℤ := sequence_b (3 * k + 1)

theorem infinite_common_elements : 
  ∀ k : ℕ, subsequence_a k = subsequence_b k :=
sorry

end infinite_common_elements_l2984_298465


namespace functional_equation_solution_l2984_298460

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + x = x * f y + f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (1/2) = 0 → f (-201) = 403 := by
  sorry

end functional_equation_solution_l2984_298460


namespace cafeteria_choices_theorem_l2984_298486

/-- Represents the number of ways to choose foods in a cafeteria -/
def cafeteriaChoices (n : ℕ) : ℕ :=
  n + 1

/-- Theorem stating that the number of ways to choose n foods in the cafeteria is n + 1 -/
theorem cafeteria_choices_theorem (n : ℕ) : 
  cafeteriaChoices n = n + 1 := by
  sorry

/-- Apples are taken in groups of 3 -/
def appleGroup : ℕ := 3

/-- Yogurts are taken in pairs -/
def yogurtPair : ℕ := 2

/-- Maximum number of bread pieces allowed -/
def maxBread : ℕ := 2

/-- Maximum number of cereal bowls allowed -/
def maxCereal : ℕ := 1

end cafeteria_choices_theorem_l2984_298486


namespace painted_cube_theorem_l2984_298413

theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end painted_cube_theorem_l2984_298413


namespace common_chord_length_l2984_298429

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end common_chord_length_l2984_298429


namespace expression_value_l2984_298495

def point_on_terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = -2 ∧ x = Real.cos α ∧ y = Real.sin α

theorem expression_value (α : Real) (h : point_on_terminal_side α) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -2 := by
  sorry

end expression_value_l2984_298495


namespace inequality_of_exponential_l2984_298411

theorem inequality_of_exponential (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/3 : ℝ)^a < (1/3 : ℝ)^b := by
  sorry

end inequality_of_exponential_l2984_298411


namespace test_questions_count_l2984_298449

theorem test_questions_count : ∀ (total_questions : ℕ),
  (total_questions % 4 = 0) →  -- Test consists of 4 equal sections
  (20 : ℝ) / total_questions > 0.60 →  -- Percentage correct > 60%
  (20 : ℝ) / total_questions < 0.70 →  -- Percentage correct < 70%
  total_questions = 32 := by
sorry

end test_questions_count_l2984_298449


namespace sequence_properties_l2984_298493

def a (i : ℕ+) : ℕ := (7^(2^i.val) - 1) / 6

theorem sequence_properties :
  ∀ i : ℕ+,
    (∀ j : ℕ+, (a (j + 1)) % (a j) = 0) ∧
    (a i) % 3 ≠ 0 ∧
    (a i) % (2^(i.val + 2)) = 0 ∧
    (a i) % (2^(i.val + 3)) ≠ 0 ∧
    ∃ p : ℕ, ∃ n : ℕ, Prime p ∧ 6 * (a i) + 1 = p^n ∧
    ∃ x y : ℕ, a i = x^2 + y^2 :=
by sorry

end sequence_properties_l2984_298493


namespace max_profit_theorem_l2984_298458

/-- Represents the profit function for a souvenir shop -/
def profit_function (x : ℝ) : ℝ := -20 * x + 3200

/-- Represents the constraint on the number of type A souvenirs -/
def constraint (x : ℝ) : Prop := x ≥ 10

/-- Theorem stating the maximum profit and the number of type A souvenirs that achieves it -/
theorem max_profit_theorem :
  ∃ (x : ℝ), constraint x ∧
  (∀ (y : ℝ), constraint y → profit_function x ≥ profit_function y) ∧
  x = 10 ∧ profit_function x = 3000 :=
sorry

end max_profit_theorem_l2984_298458


namespace min_distance_is_sqrt_2_l2984_298468

/-- Two moving lines that intersect at point A -/
structure IntersectingLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => a * x + a + b * y + 3 * b = 0
  l₂ : ℝ → ℝ → Prop := λ x y => b * x - 3 * b - a * y + a = 0

/-- The intersection point of the two lines -/
def intersectionPoint (lines : IntersectingLines) : ℝ × ℝ := sorry

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The minimum value of the length of segment OA is √2 -/
theorem min_distance_is_sqrt_2 (lines : IntersectingLines) :
  ∃ (min_dist : ℝ), ∀ (a b : ℝ),
    let lines' := { a := a, b := b : IntersectingLines }
    min_dist = Real.sqrt 2 ∧
    distance origin (intersectionPoint lines') ≥ min_dist :=
  sorry

end min_distance_is_sqrt_2_l2984_298468


namespace inequality_solution_l2984_298420

def solution_set (m : ℝ) : Set ℝ :=
  if m = -3 then { x | x > 1 }
  else if -3 < m ∧ m < -1 then { x | x < m / (m + 3) ∨ x > 1 }
  else if m < -3 then { x | 1 < x ∧ x < m / (m + 3) }
  else ∅

theorem inequality_solution (m : ℝ) (h : m < -1) :
  { x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0 } = solution_set m :=
sorry

end inequality_solution_l2984_298420


namespace mean_daily_profit_l2984_298475

theorem mean_daily_profit (days_in_month : ℕ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  days_in_month = 30 →
  first_half_mean = 225 →
  second_half_mean = 475 →
  (first_half_mean * 15 + second_half_mean * 15) / days_in_month = 350 :=
by
  sorry

end mean_daily_profit_l2984_298475


namespace absolute_value_of_c_l2984_298428

def complex_equation (a b c : ℤ) : Prop :=
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0

theorem absolute_value_of_c (a b c : ℤ) :
  complex_equation a b c →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 109 := by
  sorry

end absolute_value_of_c_l2984_298428


namespace roots_of_quadratic_l2984_298432

/-- The quadratic function f(x) = ax^2 + bx -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The equation f(x) = 6 -/
def equation (a b : ℝ) (x : ℝ) : Prop := f a b x = 6

theorem roots_of_quadratic (a b : ℝ) :
  equation a b (-2) ∧ equation a b 3 →
  (equation a b (-2) ∧ equation a b 3 ∧
   ∀ x : ℝ, equation a b x → x = -2 ∨ x = 3) :=
by sorry

end roots_of_quadratic_l2984_298432


namespace park_diameter_l2984_298415

/-- Given a circular park with a fountain, garden ring, and walking path, 
    prove that the diameter of the outer boundary is 38 feet. -/
theorem park_diameter (fountain_diameter walking_path_width garden_ring_width : ℝ) 
  (h1 : fountain_diameter = 10)
  (h2 : walking_path_width = 6)
  (h3 : garden_ring_width = 8) :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 38 := by
  sorry


end park_diameter_l2984_298415
