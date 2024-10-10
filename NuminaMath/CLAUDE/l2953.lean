import Mathlib

namespace kim_trip_time_kim_trip_time_is_120_l2953_295300

/-- The total time Kim spends away from home given the described trip conditions -/
theorem kim_trip_time : ℝ :=
  let distance_to_friend : ℝ := 30
  let detour_percentage : ℝ := 0.2
  let time_at_friends : ℝ := 30
  let speed : ℝ := 44
  let distance_back : ℝ := distance_to_friend * (1 + detour_percentage)
  let total_distance : ℝ := distance_to_friend + distance_back
  let driving_time : ℝ := total_distance / speed * 60
  driving_time + time_at_friends

theorem kim_trip_time_is_120 : kim_trip_time = 120 := by
  sorry

end kim_trip_time_kim_trip_time_is_120_l2953_295300


namespace quadratic_distinct_roots_range_l2953_295336

/-- The range of k for which the quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ 
    (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end quadratic_distinct_roots_range_l2953_295336


namespace fixed_cost_calculation_l2953_295347

/-- The fixed cost for producing products given total cost, marginal cost, and number of products. -/
theorem fixed_cost_calculation (total_cost marginal_cost : ℝ) (n : ℕ) :
  total_cost = 16000 →
  marginal_cost = 200 →
  n = 20 →
  total_cost = (marginal_cost * n) + 12000 :=
by sorry

end fixed_cost_calculation_l2953_295347


namespace no_real_solution_l2953_295372

theorem no_real_solution : ¬ ∃ (x : ℝ), (3 / (x^2 - x - 6) = 2 / (x^2 - 3*x + 2)) := by
  sorry

end no_real_solution_l2953_295372


namespace scientific_notation_of_value_l2953_295337

-- Define the nanometer to meter conversion
def nm_to_m : ℝ := 1e-9

-- Define the value in meters
def value_in_meters : ℝ := 7 * nm_to_m

-- Theorem statement
theorem scientific_notation_of_value :
  ∃ (a : ℝ) (n : ℤ), value_in_meters = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7 ∧ n = -9 :=
sorry

end scientific_notation_of_value_l2953_295337


namespace yard_area_l2953_295366

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_size : ℕ) (h1 : length = 20) (h2 : width = 18) (h3 : cut_size = 4) :
  length * width - cut_size * cut_size = 344 := by
sorry

end yard_area_l2953_295366


namespace eighth_grade_ratio_l2953_295313

theorem eighth_grade_ratio (total_students : Nat) (girls : Nat) :
  total_students = 68 →
  girls = 28 →
  (total_students - girls : Nat) / girls = 10 / 7 := by
  sorry

end eighth_grade_ratio_l2953_295313


namespace parabola_properties_l2953_295312

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the parabola D
def parabolaD (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (4, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the midpoint of PQ
def midpoint_PQ (Q : ℝ × ℝ) : Prop := (0, 0) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the angle equality
def angle_equality (A B Q : ℝ × ℝ) : Prop :=
  (A.2 - Q.2) / (A.1 - Q.1) = -(B.2 - Q.2) / (B.1 - Q.1)

-- Define the line m
def line_m (x : ℝ) : Prop := x = 3

-- Theorem statement
theorem parabola_properties :
  ∀ (A B Q : ℝ × ℝ) (k : ℝ),
  parabolaD A.1 A.2 ∧ parabolaD B.1 B.2 ∧
  line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧
  midpoint_PQ Q →
  (∀ (x y : ℝ), parabolaD x y ↔ y^2 = 4*x) ∧
  angle_equality A B Q ∧
  (∃ (x : ℝ), line_m x ∧
    ∀ (A : ℝ × ℝ), parabolaD A.1 A.2 →
    ∃ (c : ℝ), ∀ (y : ℝ), 
      (x - (A.1 + 4) / 2)^2 + (y - A.2 / 2)^2 = ((A.1 - 4)^2 + A.2^2) / 4 →
      (x - 3)^2 + y^2 = c) :=
sorry

end parabola_properties_l2953_295312


namespace circle_and_line_properties_l2953_295342

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ b : ℝ, b < 0 ∧ x^2 + (y - b)^2 = 25 ∧ 3 - b = 5

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  ∃ k : ℝ, y + 3 = k * (x + 3)

-- Define the chord length
def chord_length (x y : ℝ) : Prop :=
  ∃ (c_x c_y : ℝ), circle_C c_x c_y ∧
  ((x - c_x)^2 + (y - c_y)^2) - (((-3) - c_x)^2 + ((-3) - c_y)^2) = 20

-- Theorem statement
theorem circle_and_line_properties :
  ∀ x y : ℝ,
  circle_C x y →
  line_l x y →
  chord_length x y →
  (x^2 + (y + 2)^2 = 25) ∧
  ((x + 2*y + 9 = 0) ∨ (2*x - y + 3 = 0)) :=
by sorry

end circle_and_line_properties_l2953_295342


namespace youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l2953_295382

/-- Represents a survey scenario -/
inductive SurveyScenario
| CompanyHealthCheck
| EpidemicTemperatureCheck
| YouthEntertainment
| AirplaneSecurity

/-- Determines if a survey scenario is suitable for sampling -/
def isSuitableForSampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.YouthEntertainment => True
  | _ => False

/-- Theorem stating that the youth entertainment survey is suitable for sampling -/
theorem youth_entertainment_suitable_for_sampling :
  isSuitableForSampling SurveyScenario.YouthEntertainment :=
by sorry

/-- Theorem stating that other scenarios are not suitable for sampling -/
theorem other_scenarios_not_suitable_for_sampling (scenario : SurveyScenario) :
  scenario ≠ SurveyScenario.YouthEntertainment →
  ¬ (isSuitableForSampling scenario) :=
by sorry

end youth_entertainment_suitable_for_sampling_other_scenarios_not_suitable_for_sampling_l2953_295382


namespace circle_intersection_theorem_l2953_295377

theorem circle_intersection_theorem (O₁ O₂ T A B : ℝ × ℝ) : 
  let d := Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2)
  let r₁ := 4
  let r₂ := 6
  d ≥ 6 →
  (∃ C : ℝ × ℝ, (C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ 
               (C.1 - O₂.1)^2 + (C.2 - O₂.2)^2 = r₂^2) →
  (A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₂^2 →
  (B.1 - O₁.1)^2 + (B.2 - O₁.2)^2 = r₁^2 →
  Real.sqrt ((A.1 - T.1)^2 + (A.2 - T.2)^2) = 
    1/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  Real.sqrt ((B.1 - T.1)^2 + (B.2 - T.2)^2) = 
    2/3 * Real.sqrt ((O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2) →
  d = 6 := by
sorry

end circle_intersection_theorem_l2953_295377


namespace uncool_parents_count_l2953_295353

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 20)
  (h3 : cool_moms = 25)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 5 := by
  sorry

end uncool_parents_count_l2953_295353


namespace min_area_with_prime_dimension_l2953_295358

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Checks if at least one dimension of a rectangle is prime. -/
def hasOnePrimeDimension (r : Rectangle) : Prop := isPrime r.length ∨ isPrime r.width

/-- The main theorem stating the minimum area of a rectangle with given conditions. -/
theorem min_area_with_prime_dimension :
  ∀ r : Rectangle,
    r.length > 0 ∧ r.width > 0 →
    perimeter r = 120 →
    hasOnePrimeDimension r →
    (∀ r' : Rectangle, 
      r'.length > 0 ∧ r'.width > 0 →
      perimeter r' = 120 →
      hasOnePrimeDimension r' →
      area r ≤ area r') →
    area r = 116 :=
sorry

end min_area_with_prime_dimension_l2953_295358


namespace certain_number_proof_l2953_295384

theorem certain_number_proof :
  ∃ x : ℝ, x * (-4.5) = 2 * (-4.5) - 36 ∧ x = 10 := by
  sorry

end certain_number_proof_l2953_295384


namespace tom_next_birthday_l2953_295390

-- Define the ages as real numbers
def tom_age : ℝ := sorry
def jerry_age : ℝ := sorry
def spike_age : ℝ := sorry

-- Define the relationships between ages
axiom jerry_spike_relation : jerry_age = 1.2 * spike_age
axiom tom_jerry_relation : tom_age = 0.7 * jerry_age

-- Define the sum of ages
axiom age_sum : tom_age + jerry_age + spike_age = 36

-- Theorem to prove
theorem tom_next_birthday : ⌊tom_age⌋ + 1 = 11 := by sorry

end tom_next_birthday_l2953_295390


namespace justice_palms_l2953_295309

/-- The number of palms Justice has -/
def num_palms : ℕ := sorry

/-- The total number of plants Justice wants -/
def total_plants : ℕ := 24

/-- The number of ferns Justice has -/
def num_ferns : ℕ := 3

/-- The number of succulent plants Justice has -/
def num_succulents : ℕ := 7

/-- The number of additional plants Justice needs -/
def additional_plants : ℕ := 9

theorem justice_palms : num_palms = 5 := by sorry

end justice_palms_l2953_295309


namespace integral_points_on_line_segment_l2953_295352

def is_on_line_segment (x y : ℤ) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
  x = (22 : ℤ) + t * ((16 : ℤ) - (22 : ℤ)) ∧
  y = (12 : ℤ) + t * ((17 : ℤ) - (12 : ℤ))

theorem integral_points_on_line_segment :
  ∃! p : ℤ × ℤ, 
    is_on_line_segment p.1 p.2 ∧
    10 ≤ p.1 ∧ p.1 ≤ 30 ∧
    10 ≤ p.2 ∧ p.2 ≤ 30 :=
sorry

end integral_points_on_line_segment_l2953_295352


namespace gina_money_theorem_l2953_295394

theorem gina_money_theorem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end gina_money_theorem_l2953_295394


namespace inequality_proof_l2953_295361

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  b^2 / a < a^2 / b := by
  sorry

end inequality_proof_l2953_295361


namespace square_root_and_cube_l2953_295326

theorem square_root_and_cube (x : ℝ) (x_nonzero : x ≠ 0) :
  (Real.sqrt 144 + 3^3) / x = 39 / x :=
sorry

end square_root_and_cube_l2953_295326


namespace inequality_transformations_l2953_295321

theorem inequality_transformations :
  (∀ x : ℝ, x - 1 > 2 → x > 3) ∧
  (∀ x : ℝ, -4 * x > 8 → x < -2) := by
  sorry

end inequality_transformations_l2953_295321


namespace battery_current_l2953_295380

theorem battery_current (voltage : ℝ) (resistance : ℝ) (current : ℝ → ℝ) :
  voltage = 48 →
  (∀ R, current R = voltage / R) →
  resistance = 12 →
  current resistance = 4 :=
by sorry

end battery_current_l2953_295380


namespace sports_event_distribution_l2953_295315

/-- Represents the number of medals remaining after k days -/
def remaining_medals (k : ℕ) (m : ℕ) : ℚ :=
  if k = 0 then m
  else (6/7) * remaining_medals (k-1) m - (6/7) * k

/-- The sports event distribution problem -/
theorem sports_event_distribution (n m : ℕ) : 
  (∀ k, 1 ≤ k ∧ k < n → 
    remaining_medals k m = remaining_medals (k-1) m - (k + (1/7) * (remaining_medals (k-1) m - k))) ∧
  remaining_medals (n-1) m = n ∧
  remaining_medals n m = 0 →
  n = 6 ∧ m = 36 := by
sorry

end sports_event_distribution_l2953_295315


namespace credit_card_more_beneficial_l2953_295343

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 20000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.01

/-- Represents the annual interest rate on the debit card -/
def annual_interest_rate : ℝ := 0.06

/-- Represents the number of days in a year for interest calculation -/
def days_in_year : ℝ := 360

/-- Represents the minimum number of days for credit card to be more beneficial -/
def min_days : ℕ := 31

theorem credit_card_more_beneficial :
  ∀ N : ℕ,
  N ≥ min_days →
  (purchase_amount * credit_cashback_rate) + 
  (purchase_amount * annual_interest_rate * N / days_in_year) >
  purchase_amount * debit_cashback_rate :=
sorry

end credit_card_more_beneficial_l2953_295343


namespace turtle_conservation_l2953_295374

theorem turtle_conservation (G H L : ℕ) : 
  G = 800 → H = 2 * G → L = 3 * G → G + H + L = 4800 := by
  sorry

end turtle_conservation_l2953_295374


namespace trigonometric_expression_equality_l2953_295320

theorem trigonometric_expression_equality (x : Real) :
  (x > π / 2 ∧ x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3 * Real.tan x - 4 = 0 →
  (Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 1 / 3 := by
  sorry

end trigonometric_expression_equality_l2953_295320


namespace quadratic_solution_range_l2953_295399

theorem quadratic_solution_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 := by
sorry

end quadratic_solution_range_l2953_295399


namespace triangle_area_l2953_295354

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  A = 30 * π / 180 →
  B = 60 * π / 180 →
  C = π - A - B →
  b = a * Real.sin B / Real.sin A →
  (1/2) * a * b * Real.sin C = (9 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l2953_295354


namespace probability_shaded_is_two_fifths_l2953_295318

/-- A structure representing the triangle selection scenario -/
structure TriangleSelection where
  total_triangles : ℕ
  shaded_triangles : ℕ
  shaded_triangles_le_total : shaded_triangles ≤ total_triangles

/-- The probability of selecting a triangle with a shaded part -/
def probability_shaded (ts : TriangleSelection) : ℚ :=
  ts.shaded_triangles / ts.total_triangles

/-- Theorem stating that the probability of selecting a shaded triangle is 2/5 -/
theorem probability_shaded_is_two_fifths (ts : TriangleSelection) 
    (h1 : ts.total_triangles = 5)
    (h2 : ts.shaded_triangles = 2) : 
  probability_shaded ts = 2 / 5 := by
  sorry

end probability_shaded_is_two_fifths_l2953_295318


namespace same_color_probability_l2953_295348

/-- The probability of drawing two balls of the same color from a bag containing 4 green balls and 8 white balls. -/
theorem same_color_probability (green : ℕ) (white : ℕ) (h1 : green = 4) (h2 : white = 8) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := (p_green * (green - 1) / (total - 1)) + (p_white * (white - 1) / (total - 1))
  p_same_color = 17 / 33 := by
  sorry

end same_color_probability_l2953_295348


namespace equation_solutions_l2953_295363

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1) ∧
  (∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ x = (3 + Real.sqrt 17) / 4 ∨ x = (3 - Real.sqrt 17) / 4) :=
by sorry

end equation_solutions_l2953_295363


namespace point_not_on_transformed_plane_l2953_295331

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -2, y := 4, z := 1 }
  let a : Plane := { a := 3, b := 1, c := 2, d := 2 }
  let k : ℝ := 3
  let a' := transformPlane a k
  ¬ pointOnPlane A a' := by sorry

end point_not_on_transformed_plane_l2953_295331


namespace connor_sleep_time_l2953_295305

theorem connor_sleep_time (puppy_sleep : ℕ) (luke_sleep : ℕ) (connor_sleep : ℕ) : 
  puppy_sleep = 16 →
  puppy_sleep = 2 * luke_sleep →
  luke_sleep = connor_sleep + 2 →
  connor_sleep = 6 :=
by sorry

end connor_sleep_time_l2953_295305


namespace chess_tournament_draws_l2953_295381

/-- Represents a chess tournament with a fixed number of participants. -/
structure ChessTournament where
  n : ℕ  -- number of participants
  lists : Fin n → Fin 12 → Set (Fin n)  -- lists[i][j] is the jth list of participant i
  
  list_rule_1 : ∀ i, lists i 0 = {i}
  list_rule_2 : ∀ i j, j > 0 → lists i j ⊇ lists i (j-1)
  list_rule_12 : ∀ i, lists i 11 ≠ lists i 10

/-- The number of draws in the tournament. -/
def num_draws (t : ChessTournament) : ℕ :=
  (t.n.choose 2) - t.n

theorem chess_tournament_draws (t : ChessTournament) (h : t.n = 12) : 
  num_draws t = 54 := by
  sorry


end chess_tournament_draws_l2953_295381


namespace number_satisfying_equation_l2953_295369

theorem number_satisfying_equation : ∃ x : ℝ, (0.08 * x) + (0.10 * 40) = 5.92 ∧ x = 24 := by
  sorry

end number_satisfying_equation_l2953_295369


namespace color_drawing_percentage_increase_l2953_295396

def black_and_white_cost : ℝ := 160
def color_cost : ℝ := 240

theorem color_drawing_percentage_increase : 
  (color_cost - black_and_white_cost) / black_and_white_cost * 100 = 50 := by
  sorry

end color_drawing_percentage_increase_l2953_295396


namespace unique_three_digit_divisible_by_nine_l2953_295319

theorem unique_three_digit_divisible_by_nine : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 4 ∧          -- units digit is 4
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 9 = 0 ∧           -- divisible by 9
    n = 684               -- the number is 684
  := by sorry

end unique_three_digit_divisible_by_nine_l2953_295319


namespace cone_sphere_volume_ratio_l2953_295317

/-- Theorem: Ratio of cone height to base radius when cone volume is one-third of sphere volume -/
theorem cone_sphere_volume_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 : ℝ) * ((4 / 3 : ℝ) * Real.pi * r^3) = (1 / 3 : ℝ) * Real.pi * r^2 * h → h / r = 4 / 3 :=
by sorry

end cone_sphere_volume_ratio_l2953_295317


namespace existence_of_relatively_prime_divisible_combination_l2953_295395

theorem existence_of_relatively_prime_divisible_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := by
  sorry

end existence_of_relatively_prime_divisible_combination_l2953_295395


namespace product_sum_theorem_l2953_295346

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a + b + c = 22) : 
  a*b + b*c + a*c = 131 := by sorry

end product_sum_theorem_l2953_295346


namespace josh_remaining_money_l2953_295338

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℝ) (hat_cost : ℝ) (pencil_cost : ℝ) (cookie_cost : ℝ) (num_cookies : ℕ) : ℝ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * num_cookies)

/-- Proves that Josh has $3 left after his purchases -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end josh_remaining_money_l2953_295338


namespace pizza_pieces_per_pizza_pizza_pieces_theorem_l2953_295389

theorem pizza_pieces_per_pizza 
  (num_students : ℕ) 
  (pizzas_per_student : ℕ) 
  (total_pieces : ℕ) : ℕ :=
  let total_pizzas := num_students * pizzas_per_student
  total_pieces / total_pizzas

#check pizza_pieces_per_pizza 10 20 1200 = 6

-- Proof
theorem pizza_pieces_theorem :
  pizza_pieces_per_pizza 10 20 1200 = 6 := by
  sorry

end pizza_pieces_per_pizza_pizza_pieces_theorem_l2953_295389


namespace john_school_year_hours_l2953_295301

/-- Calculates the required working hours per week during school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) : ℚ :=
  let hourly_wage : ℚ := summer_earnings / (summer_weeks * summer_hours_per_week)
  let total_school_year_hours : ℚ := school_year_target / hourly_wage
  total_school_year_hours / school_year_weeks

theorem john_school_year_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_target : ℕ) 
  (h1 : summer_weeks = 8) 
  (h2 : summer_hours_per_week = 40) 
  (h3 : summer_earnings = 4000) 
  (h4 : school_year_weeks = 25) 
  (h5 : school_year_target = 5000) :
  school_year_hours_per_week summer_weeks summer_hours_per_week summer_earnings school_year_weeks school_year_target = 16 := by
  sorry

end john_school_year_hours_l2953_295301


namespace set_operations_l2953_295345

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B : Set ℝ := {x | -1 < x ∧ x < 5}

theorem set_operations :
  (A ∪ B = {x | -3 ≤ x ∧ x < 5}) ∧
  (A ∩ B = {x | -1 < x ∧ x ≤ 4}) ∧
  ((U \ A) ∩ B = {x | 4 < x ∧ x < 5}) := by
  sorry

end set_operations_l2953_295345


namespace circumcenter_from_equal_distances_l2953_295360

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A triangle in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

/-- Check if a point is outside a plane -/
def isOutsidePlane (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a line is perpendicular to a plane -/
def isPerpendicularToPlane (P O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Check if a point is the foot of a perpendicular -/
def isFootOfPerpendicular (O : Point3D) (P : Point3D) (t : Triangle3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (P Q : Point3D) : ℝ := sorry

/-- Check if a point is the circumcenter of a triangle -/
def isCircumcenter (O : Point3D) (t : Triangle3D) : Prop := sorry

/-- Main theorem -/
theorem circumcenter_from_equal_distances (P O : Point3D) (t : Triangle3D) :
  isOutsidePlane P t →
  isPerpendicularToPlane P O t →
  isFootOfPerpendicular O P t →
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C →
  isCircumcenter O t := by sorry

end circumcenter_from_equal_distances_l2953_295360


namespace loan_amount_proof_l2953_295383

/-- Represents a loan with simple interest -/
structure Loan where
  principal : ℕ
  rate : ℕ
  time : ℕ
  interest : ℕ

/-- Calculates the simple interest for a loan -/
def simpleInterest (l : Loan) : ℕ :=
  l.principal * l.rate * l.time / 100

theorem loan_amount_proof (l : Loan) :
  l.rate = 8 ∧ l.time = l.rate ∧ l.interest = 704 →
  simpleInterest l = l.interest →
  l.principal = 1100 := by
sorry

end loan_amount_proof_l2953_295383


namespace quadratic_properties_l2953_295329

/-- A quadratic function passing through (3, -1) -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

theorem quadratic_properties (a b : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b
  -- The function passes through (3, -1)
  (f 3 = -1) →
  -- (2, 2-2a) does not lie on the graph
  (f 2 ≠ 2 - 2*a) ∧
  -- When the graph intersects the x-axis at only one point
  (∃ x : ℝ, (f x = 0 ∧ ∀ y : ℝ, f y = 0 → y = x)) →
  -- The function is either y = -x^2 + 4x - 4 or y = -1/9x^2 + 4/3x - 4
  ((a = -1 ∧ b = 4) ∨ (a = -1/9 ∧ b = 4/3)) ∧
  -- When the graph passes through points (x₁, y₁) and (x₂, y₂) with x₁ < x₂ ≤ 2/3 and y₁ > y₂
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → x₂ ≤ 2/3 → f x₁ = y₁ → f x₂ = y₂ → y₁ > y₂ → a ≥ 3/5) :=
by sorry

end quadratic_properties_l2953_295329


namespace total_salary_is_616_l2953_295379

/-- The salary of employee N in dollars per week -/
def salary_N : ℝ := 280

/-- The ratio of M's salary to N's salary -/
def salary_ratio : ℝ := 1.2

/-- The salary of employee M in dollars per week -/
def salary_M : ℝ := salary_ratio * salary_N

/-- The total amount paid to both employees per week -/
def total_salary : ℝ := salary_M + salary_N

theorem total_salary_is_616 : total_salary = 616 := by
  sorry

end total_salary_is_616_l2953_295379


namespace doughnuts_per_person_l2953_295306

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts with friends -/
theorem doughnuts_per_person :
  ∀ (samuel_dozens cathy_dozens num_friends : ℕ),
  samuel_dozens = 2 →
  cathy_dozens = 3 →
  num_friends = 8 →
  (samuel_dozens * 12 + cathy_dozens * 12) / (num_friends + 2) = 6 :=
by sorry

end doughnuts_per_person_l2953_295306


namespace g_of_two_equals_six_l2953_295334

/-- Given a function g where g(x) = 5x - 4 for all x, prove that g(2) = 6 -/
theorem g_of_two_equals_six (g : ℝ → ℝ) (h : ∀ x, g x = 5 * x - 4) : g 2 = 6 := by
  sorry

end g_of_two_equals_six_l2953_295334


namespace complement_union_theorem_l2953_295388

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by sorry

end complement_union_theorem_l2953_295388


namespace min_k_value_l2953_295376

-- Define the function f(x) = x(ln x + 1) / (x - 2)
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1) / (x - 2)

-- State the theorem
theorem min_k_value : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ ∃ k : ℕ, k > 0 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) → 
  (∀ k : ℕ, k > 0 → (∃ x : ℝ, x > 2 ∧ k * (x - 2) > x * (Real.log x + 1)) → k ≥ 5) ∧
  (∃ x : ℝ, x > 2 ∧ 5 * (x - 2) > x * (Real.log x + 1)) :=
sorry

end min_k_value_l2953_295376


namespace divisibility_implies_sum_product_l2953_295362

theorem divisibility_implies_sum_product (p q r s : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s = 
    (x^4 + 4*x^3 + 6*x^2 + 4*x + 1) * k) →
  (p + q + r) * s = -2.2 := by
sorry

end divisibility_implies_sum_product_l2953_295362


namespace alcohol_solution_concentration_l2953_295332

/-- Proves that adding 1.8 liters of pure alcohol to a 6-liter solution
    that is 35% alcohol results in a 50% alcohol solution. -/
theorem alcohol_solution_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_alcohol : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.35)
  (h3 : added_alcohol = 1.8)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end alcohol_solution_concentration_l2953_295332


namespace abs_neg_three_l2953_295393

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l2953_295393


namespace no_divisible_by_ten_l2953_295344

/-- The function g(x) = x^2 + 5x + 3 -/
def g (x : ℤ) : ℤ := x^2 + 5*x + 3

/-- The set T of integers from 0 to 30 -/
def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

/-- Theorem: There are no integers t in T such that g(t) is divisible by 10 -/
theorem no_divisible_by_ten : ∀ t ∈ T, ¬(g t % 10 = 0) := by sorry

end no_divisible_by_ten_l2953_295344


namespace hash_eight_three_l2953_295350

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_eight_three : hash 8 3 = 127 :=
  by
    sorry

-- Define the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + 2 * s + 1

end hash_eight_three_l2953_295350


namespace square_root_problem_l2953_295340

theorem square_root_problem (x : ℝ) (a : ℝ) 
  (h1 : x > 0)
  (h2 : Real.sqrt x = 3 * a - 4)
  (h3 : Real.sqrt x = 1 - 6 * a) :
  a = -1 ∧ x = 49 := by
  sorry

end square_root_problem_l2953_295340


namespace necessary_not_sufficient_condition_l2953_295368

-- Define the proposition
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z : ℝ, x > y ∧ x * z^2 ≤ y * z^2) ∧
  (∀ x y z : ℝ, x * z^2 > y * z^2 → x > y) :=
by sorry

end necessary_not_sufficient_condition_l2953_295368


namespace vector_linear_combination_l2953_295303

/-- Given two vectors in ℝ², prove that their linear combination results in the expected vector. -/
theorem vector_linear_combination (a b : ℝ × ℝ) (h1 : a = (-1, 0)) (h2 : b = (0, 2)) :
  (2 : ℝ) • a - (3 : ℝ) • b = (-2, -6) := by sorry

end vector_linear_combination_l2953_295303


namespace machine_purchase_price_l2953_295322

/-- Given a machine with specified costs and selling price, calculates the original purchase price. -/
theorem machine_purchase_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percentage : ℚ) (selling_price : ℕ) : 
  repair_cost = 5000 →
  transport_cost = 1000 →
  profit_percentage = 50 / 100 →
  selling_price = 25500 →
  ∃ (purchase_price : ℕ), 
    (purchase_price : ℚ) + repair_cost + transport_cost = 
      selling_price / (1 + profit_percentage) ∧
    purchase_price = 11000 :=
by sorry

end machine_purchase_price_l2953_295322


namespace train_crossing_time_l2953_295355

/-- Proves that the time taken for the first train to cross a telegraph post is 10 seconds,
    given the conditions of the problem. -/
theorem train_crossing_time
  (train_length : ℝ)
  (second_train_time : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : second_train_time = 15)
  (h3 : crossing_time = 12) :
  let second_train_speed := train_length / second_train_time
  let relative_speed := 2 * train_length / crossing_time
  let first_train_speed := relative_speed - second_train_speed
  train_length / first_train_speed = 10 :=
by sorry


end train_crossing_time_l2953_295355


namespace regular_polygon_diagonals_sides_l2953_295387

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℚ := n * (n - 3) / 2

/-- Theorem: A regular polygon whose number of diagonals is three times its number of sides has 9 sides -/
theorem regular_polygon_diagonals_sides : ∃ n : ℕ, n > 2 ∧ num_diagonals n = 3 * n ∧ n = 9 := by
  sorry

end regular_polygon_diagonals_sides_l2953_295387


namespace mashed_potatoes_count_l2953_295392

theorem mashed_potatoes_count (tomatoes bacon : ℕ) 
  (h1 : tomatoes = 79)
  (h2 : bacon = 467) : 
  ∃ mashed_potatoes : ℕ, mashed_potatoes = tomatoes + 65 ∧ mashed_potatoes = 144 :=
by
  sorry

end mashed_potatoes_count_l2953_295392


namespace quadratic_equation_real_solutions_l2953_295357

theorem quadratic_equation_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) := by
  sorry

end quadratic_equation_real_solutions_l2953_295357


namespace constant_speed_travel_time_l2953_295330

/-- 
Given:
- A person drives 120 miles in 3 hours
- The person maintains the same speed for another trip of 200 miles
Prove: The second trip will take 5 hours
-/
theorem constant_speed_travel_time 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) 
  (h1 : distance1 = 120) 
  (h2 : time1 = 3) 
  (h3 : distance2 = 200) : 
  (distance2 / (distance1 / time1)) = 5 := by
  sorry

#check constant_speed_travel_time

end constant_speed_travel_time_l2953_295330


namespace f_monotone_decreasing_l2953_295364

/-- A piecewise function f: ℝ → ℝ defined by two parts -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (2*a - 1)*x + 7*a - 2 else a^x

/-- Theorem stating the condition for f to be monotonically decreasing -/
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ 3/8 ≤ a ∧ a < 1/2 :=
sorry

end f_monotone_decreasing_l2953_295364


namespace tan_point_zero_l2953_295310

theorem tan_point_zero (φ : ℝ) : 
  (fun x => Real.tan (x + φ)) (π / 3) = 0 → φ = -π / 3 := by
  sorry

end tan_point_zero_l2953_295310


namespace pure_imaginary_product_l2953_295365

theorem pure_imaginary_product (m : ℝ) : 
  (∃ (z : ℂ), z * z = -1 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).re = 0 ∧ (Complex.mk 2 (-m) * Complex.mk 1 (-1)).im ≠ 0) → 
  m = 2 := by
sorry

end pure_imaginary_product_l2953_295365


namespace unshaded_parts_sum_l2953_295324

theorem unshaded_parts_sum (square_area shaded_area : ℝ) 
  (h1 : square_area = 36) 
  (h2 : shaded_area = 27) 
  (p q r s : ℝ) :
  p + q + r + s = 9 := by sorry

end unshaded_parts_sum_l2953_295324


namespace sum_a_t_equals_41_l2953_295302

theorem sum_a_t_equals_41 (a t : ℝ) (ha : a > 0) (ht : t > 0) 
  (h : Real.sqrt (6 + a / t) = 6 * Real.sqrt (a / t)) : a + t = 41 :=
sorry

end sum_a_t_equals_41_l2953_295302


namespace last_two_digits_sum_l2953_295385

theorem last_two_digits_sum (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end last_two_digits_sum_l2953_295385


namespace train_speed_excluding_stoppages_l2953_295356

/-- Given a train that travels 40 km in one hour (including stoppages) and stops for 20 minutes each hour, 
    its speed excluding stoppages is 60 kmph. -/
theorem train_speed_excluding_stoppages : 
  ∀ (speed_with_stops : ℝ) (stop_time : ℝ) (total_time : ℝ),
  speed_with_stops = 40 →
  stop_time = 20 →
  total_time = 60 →
  (total_time - stop_time) / total_time * speed_with_stops = 60 := by
sorry

end train_speed_excluding_stoppages_l2953_295356


namespace sqrt_x_minus_y_equals_plus_minus_two_l2953_295333

theorem sqrt_x_minus_y_equals_plus_minus_two
  (x y : ℝ) 
  (h : Real.sqrt (x - 3) + 2 * abs (y + 1) = 0) :
  Real.sqrt (x - y) = 2 ∨ Real.sqrt (x - y) = -2 :=
sorry

end sqrt_x_minus_y_equals_plus_minus_two_l2953_295333


namespace book_arrangement_l2953_295311

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  (n! / k!) = 840 := by
  sorry

end book_arrangement_l2953_295311


namespace daniels_age_l2953_295370

/-- Given the ages of Uncle Ben, Edward, and Daniel, prove Daniel's age --/
theorem daniels_age (uncle_ben_age : ℚ) (edward_age : ℚ) (daniel_age : ℚ) : 
  uncle_ben_age = 50 →
  edward_age = 2/3 * uncle_ben_age →
  daniel_age = edward_age - 7 →
  daniel_age = 79/3 := by sorry

end daniels_age_l2953_295370


namespace stone_breadth_proof_l2953_295308

/-- Given a hall and stones with specified dimensions, prove the breadth of each stone -/
theorem stone_breadth_proof (hall_length hall_width : ℝ) (stone_length : ℝ) (num_stones : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.3)
  (h4 : num_stones = 3600) :
  ∃ (stone_breadth : ℝ), 
    stone_breadth = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_length * stone_breadth * num_stones) :=
by sorry

end stone_breadth_proof_l2953_295308


namespace profit_equation_correct_l2953_295335

/-- Represents the profit scenario for a product with varying price and sales volume. -/
def profit_equation (x : ℝ) : Prop :=
  let initial_purchase_price : ℝ := 35
  let initial_selling_price : ℝ := 40
  let initial_sales_volume : ℝ := 200
  let price_increase : ℝ := x
  let sales_volume_decrease : ℝ := 5 * x
  let new_profit_per_unit : ℝ := (initial_selling_price + price_increase) - initial_purchase_price
  let new_sales_volume : ℝ := initial_sales_volume - sales_volume_decrease
  let total_profit : ℝ := 1870
  (new_profit_per_unit * new_sales_volume) = total_profit

/-- Theorem stating that the given equation correctly represents the profit scenario. -/
theorem profit_equation_correct :
  ∀ x : ℝ, profit_equation x ↔ (x + 5) * (200 - 5 * x) = 1870 :=
sorry

end profit_equation_correct_l2953_295335


namespace hotel_towels_l2953_295373

theorem hotel_towels (rooms : ℕ) (people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : rooms = 20)
  (h2 : people_per_room = 5)
  (h3 : towels_per_person = 3) :
  rooms * people_per_room * towels_per_person = 300 :=
by sorry

end hotel_towels_l2953_295373


namespace mary_gave_three_green_crayons_l2953_295314

/-- The number of green crayons Mary gave to Becky -/
def green_crayons_given : ℕ := 3

/-- The initial number of green crayons Mary had -/
def initial_green : ℕ := 5

/-- The initial number of blue crayons Mary had -/
def initial_blue : ℕ := 8

/-- The number of blue crayons Mary gave away -/
def blue_given : ℕ := 1

/-- The number of crayons Mary has left -/
def crayons_left : ℕ := 9

theorem mary_gave_three_green_crayons :
  green_crayons_given = initial_green - (initial_green + initial_blue - blue_given - crayons_left) :=
by sorry

end mary_gave_three_green_crayons_l2953_295314


namespace students_in_both_band_and_chorus_l2953_295386

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 30 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 110 140 220 = 30 := by
  sorry

end students_in_both_band_and_chorus_l2953_295386


namespace absolute_value_equation_product_l2953_295398

theorem absolute_value_equation_product (x : ℝ) :
  (∀ x, |x - 5| - 4 = 3 → x = 12 ∨ x = -2) ∧
  (∃ x₁ x₂, |x₁ - 5| - 4 = 3 ∧ |x₂ - 5| - 4 = 3 ∧ x₁ * x₂ = -24) :=
by sorry

end absolute_value_equation_product_l2953_295398


namespace soccer_handshakes_l2953_295328

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) : 
  team_size = 11 → referee_count = 3 → 
  (team_size * team_size) + (2 * team_size * referee_count) = 187 := by
  sorry

#check soccer_handshakes

end soccer_handshakes_l2953_295328


namespace triple_tangent_identity_l2953_295371

theorem triple_tangent_identity (x y z : ℝ) (h : x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) =
  ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) := by
  sorry

end triple_tangent_identity_l2953_295371


namespace circle_min_area_l2953_295325

/-- Given positive real numbers x and y satisfying the equation 3/(2+x) + 3/(2+y) = 1,
    this theorem states that (x-4)^2 + (y-4)^2 = 256 is the equation of the circle
    with center (x,y) and radius xy when its area is minimized. -/
theorem circle_min_area (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : 3 / (2 + x) + 3 / (2 + y) = 1) :
    ∃ (center_x center_y : ℝ),
      (x - center_x)^2 + (y - center_y)^2 = (x * y)^2 ∧
      center_x = 4 ∧ center_y = 4 ∧ x * y = 16 ∧
      ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / (2 + x') + 3 / (2 + y') = 1 →
        x' * y' ≥ 16 := by
  sorry

end circle_min_area_l2953_295325


namespace chess_tournament_participants_l2953_295375

theorem chess_tournament_participants (total_games : ℕ) (h : total_games = 231) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 22 ∧ n - 1 = 21 := by
sorry

end chess_tournament_participants_l2953_295375


namespace probability_one_each_color_l2953_295367

def total_marbles : ℕ := 7
def red_marbles : ℕ := 2
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def yellow_marbles : ℕ := 1
def marbles_drawn : ℕ := 3

theorem probability_one_each_color (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) (yellow : ℕ) (drawn : ℕ)
  (h1 : total = red + blue + green + yellow)
  (h2 : drawn = 3)
  (h3 : red = 2)
  (h4 : blue = 2)
  (h5 : green = 2)
  (h6 : yellow = 1) :
  (red * blue * green : ℚ) / Nat.choose total drawn = 8 / 35 :=
by sorry

end probability_one_each_color_l2953_295367


namespace bicycle_speed_problem_l2953_295378

/-- Proves that given a total distance of 350 km, where the first 200 km is traveled at 20 km/h,
    and the average speed for the entire trip is 17.5 km/h, the speed for the remaining distance is 15 km/h. -/
theorem bicycle_speed_problem (total_distance : ℝ) (first_part_distance : ℝ) (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 350 →
  first_part_distance = 200 →
  first_part_speed = 20 →
  average_speed = 17.5 →
  (total_distance - first_part_distance) / ((total_distance / average_speed) - (first_part_distance / first_part_speed)) = 15 :=
by sorry

end bicycle_speed_problem_l2953_295378


namespace nathan_tokens_l2953_295316

/-- Calculates the total number of tokens used by Nathan at the arcade -/
def total_tokens (air_hockey_games basketball_games skee_ball_games shooting_games racing_games : ℕ)
  (air_hockey_cost basketball_cost skee_ball_cost shooting_cost racing_cost : ℕ) : ℕ :=
  air_hockey_games * air_hockey_cost +
  basketball_games * basketball_cost +
  skee_ball_games * skee_ball_cost +
  shooting_games * shooting_cost +
  racing_games * racing_cost

theorem nathan_tokens :
  total_tokens 7 12 9 6 5 6 8 4 7 5 = 241 := by
  sorry

end nathan_tokens_l2953_295316


namespace matrix_power_4_l2953_295323

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_4 :
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end matrix_power_4_l2953_295323


namespace georginas_parrot_learning_rate_l2953_295304

/-- The number of phrases Georgina's parrot knows now -/
def current_phrases : ℕ := 17

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_owned : ℕ := 49

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases Georgina teaches her parrot per week -/
def phrases_per_week : ℚ :=
  (current_phrases - initial_phrases) / (days_owned / days_per_week)

theorem georginas_parrot_learning_rate :
  phrases_per_week = 2 := by sorry

end georginas_parrot_learning_rate_l2953_295304


namespace power_function_not_through_origin_l2953_295307

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  (m = 1 ∨ m = 2) := by
  sorry

end power_function_not_through_origin_l2953_295307


namespace tangent_point_coordinates_l2953_295327

-- Define the set G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.2 ∧ p.2 ≤ 8 ∧ (p.1 - 3)^2 + 31 = (p.2 - 4)^2 + 8 * Real.sqrt (p.2 * (8 - p.2))}

-- Define the tangent line condition
def isTangentLine (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ p ∈ G ∧
  ∀ q : ℝ × ℝ, q ∈ G → q.2 ≤ m * q.1 + b

-- Theorem statement
theorem tangent_point_coordinates :
  ∃! p : ℝ × ℝ, p ∈ G ∧ 
    ∃ m : ℝ, m < 0 ∧ 
      isTangentLine m 4 p ∧
      p = (12/5, 8/5) := by
  sorry

end tangent_point_coordinates_l2953_295327


namespace smallest_wonder_number_l2953_295391

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for a "wonder number" -/
def is_wonder_number (n : ℕ) : Prop :=
  (digit_sum n = digit_sum (3 * n)) ∧ 
  (digit_sum n ≠ digit_sum (2 * n))

/-- Theorem stating that 144 is the smallest wonder number -/
theorem smallest_wonder_number : 
  is_wonder_number 144 ∧ ∀ n < 144, ¬is_wonder_number n := by sorry

end smallest_wonder_number_l2953_295391


namespace hemisphere_to_sphere_surface_area_l2953_295339

/-- Given a hemisphere with base area 81π, prove that the total surface area
    of the sphere obtained by adding a top circular lid is 324π. -/
theorem hemisphere_to_sphere_surface_area :
  ∀ r : ℝ,
  r > 0 →
  π * r^2 = 81 * π →
  4 * π * r^2 = 324 * π := by
sorry

end hemisphere_to_sphere_surface_area_l2953_295339


namespace simplify_expression_l2953_295349

theorem simplify_expression :
  3 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 3 - 2 * Real.sqrt 5 := by
  sorry

end simplify_expression_l2953_295349


namespace shortest_altitude_of_triangle_l2953_295359

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 7.2 := by sorry

end shortest_altitude_of_triangle_l2953_295359


namespace reciprocal_of_sqrt_two_l2953_295341

theorem reciprocal_of_sqrt_two : Real.sqrt 2 * (Real.sqrt 2 / 2) = 1 := by
  sorry

end reciprocal_of_sqrt_two_l2953_295341


namespace intersection_of_A_and_B_l2953_295351

def A : Set ℝ := {x | x^2 - 1 ≤ 0}
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l2953_295351


namespace fraction_evaluation_l2953_295397

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end fraction_evaluation_l2953_295397
