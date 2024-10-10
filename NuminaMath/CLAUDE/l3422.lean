import Mathlib

namespace apple_basket_problem_l3422_342257

theorem apple_basket_problem (n : ℕ) : 
  (27 * n > 25 * n) ∧                  -- A has more apples than B
  (27 * n - 4 < 25 * n + 4) ∧          -- Moving 4 apples makes B have more
  (27 * n - 3 ≥ 25 * n + 3) →          -- Moving 3 apples doesn't make B have more
  27 * n + 25 * n = 156 :=              -- Total number of apples
by sorry

end apple_basket_problem_l3422_342257


namespace f_monotone_decreasing_l3422_342262

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x^2 - 3*x

-- State the theorem
theorem f_monotone_decreasing : 
  ∀ (x y : ℝ), x < y → f x > f y :=
by
  sorry

end f_monotone_decreasing_l3422_342262


namespace triangle_abc_properties_l3422_342200

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ (1/2 : ℝ) * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3 / 4 := by
  sorry

end triangle_abc_properties_l3422_342200


namespace ice_cream_ratio_l3422_342258

theorem ice_cream_ratio (sunday : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) 
  (h1 : sunday = 4)
  (h2 : monday = 3 * sunday)
  (h3 : tuesday = monday / 3)
  (h4 : wednesday = 18)
  (h5 : sunday + monday + tuesday = wednesday + (sunday + monday + tuesday - wednesday)) :
  (sunday + monday + tuesday - wednesday) / tuesday = 1 / 2 := by
sorry

end ice_cream_ratio_l3422_342258


namespace smallest_pyramid_height_approx_l3422_342240

/-- Represents a square-based pyramid with a cylinder inside. -/
structure PyramidWithCylinder where
  base_side_length : ℝ
  cylinder_diameter : ℝ
  cylinder_length : ℝ

/-- Calculates the smallest possible height of the pyramid given its configuration. -/
def smallest_pyramid_height (p : PyramidWithCylinder) : ℝ :=
  sorry

/-- The theorem stating the smallest possible height of the pyramid. -/
theorem smallest_pyramid_height_approx :
  let p := PyramidWithCylinder.mk 20 10 10
  ∃ ε > 0, abs (smallest_pyramid_height p - 22.1) < ε :=
sorry

end smallest_pyramid_height_approx_l3422_342240


namespace determinant_positive_range_l3422_342215

def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_positive_range (x : ℝ) :
  second_order_determinant 2 (3 - x) 1 x > 0 ↔ x > 1 :=
by sorry

end determinant_positive_range_l3422_342215


namespace line_equation_proof_l3422_342225

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = 4x + 3 -/
def given_line : Line :=
  { slope := 4, intercept := 3 }

/-- The point (1, 1) -/
def point : (ℝ × ℝ) :=
  (1, 1)

theorem line_equation_proof :
  ∃ (l : Line),
    parallel l given_line ∧
    passes_through l point.1 point.2 ∧
    l.slope = 4 ∧
    l.intercept = -3 := by
  sorry

end line_equation_proof_l3422_342225


namespace coffee_bean_price_proof_l3422_342288

/-- The price of the first type of coffee bean -/
def first_bean_price : ℝ := 33

/-- The price of the second type of coffee bean -/
def second_bean_price : ℝ := 12

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price_per_pound : ℝ := 11.25

/-- The weight of each type of bean used in the mixture -/
def each_bean_weight : ℝ := 25

theorem coffee_bean_price_proof : 
  first_bean_price * each_bean_weight + 
  second_bean_price * each_bean_weight = 
  total_mixture_weight * mixture_price_per_pound :=
by sorry

end coffee_bean_price_proof_l3422_342288


namespace max_pq_plus_r_for_primes_l3422_342281

theorem max_pq_plus_r_for_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → 
  p * q + q * r + r * p = 2016 → 
  p * q + r ≤ 1008 := by
sorry

end max_pq_plus_r_for_primes_l3422_342281


namespace prob_not_late_prob_late_and_miss_bus_l3422_342289

-- Define the probability of Sam being late
def prob_late : ℚ := 5/9

-- Define the probability of Sam missing the bus if late
def prob_miss_bus_if_late : ℚ := 1/3

-- Theorem 1: Probability that Sam is not late
theorem prob_not_late : 1 - prob_late = 4/9 := by sorry

-- Theorem 2: Probability that Sam is late and misses the bus
theorem prob_late_and_miss_bus : prob_late * prob_miss_bus_if_late = 5/27 := by sorry

end prob_not_late_prob_late_and_miss_bus_l3422_342289


namespace one_third_recipe_sugar_l3422_342279

def original_recipe : ℚ := 7 + 3/4

theorem one_third_recipe_sugar (original : ℚ) (reduced : ℚ) : 
  original = 7 + 3/4 → reduced = original * (1/3) → reduced = 2 + 7/12 := by
  sorry

end one_third_recipe_sugar_l3422_342279


namespace gcd_equation_solution_l3422_342295

theorem gcd_equation_solution (b d : ℕ) : 
  Nat.gcd b 175 = d → 
  176 * (b - 11 * d + 1) = 5 * d + 1 → 
  b = 385 := by
sorry

end gcd_equation_solution_l3422_342295


namespace particle_max_elevation_l3422_342211

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 20

/-- The maximum elevation achieved by the particle -/
def max_elevation : ℝ := 520

/-- Theorem stating that the maximum elevation is 520 feet -/
theorem particle_max_elevation :
  ∃ t : ℝ, elevation t = max_elevation ∧ ∀ u : ℝ, elevation u ≤ max_elevation := by
  sorry

end particle_max_elevation_l3422_342211


namespace sqrt_nine_plus_sixteen_l3422_342231

theorem sqrt_nine_plus_sixteen : Real.sqrt (9 + 16) = 5 := by sorry

end sqrt_nine_plus_sixteen_l3422_342231


namespace softball_opponent_score_l3422_342236

theorem softball_opponent_score :
  let team_scores : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let num_games := team_scores.length
  let num_losses := (team_scores.filter (λ x => x % 2 = 1)).length
  let opponent_scores_losses := (team_scores.filter (λ x => x % 2 = 1)).map (λ x => x + 1)
  let opponent_scores_wins := (team_scores.filter (λ x => x % 2 = 0)).map (λ x => x / 2)
  num_games = 10 →
  num_losses = 5 →
  opponent_scores_losses.sum + opponent_scores_wins.sum = 45 :=
by sorry

end softball_opponent_score_l3422_342236


namespace power_division_l3422_342263

theorem power_division (x : ℝ) : x^3 / x^2 = x := by
  sorry

end power_division_l3422_342263


namespace difference_of_squares_l3422_342283

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l3422_342283


namespace functions_and_tangent_line_l3422_342272

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x
def g (b c : ℝ) (x : ℝ) : ℝ := b * x^2 + c

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- State the theorem
theorem functions_and_tangent_line 
  (a b c : ℝ) 
  (h1 : f a P.1 = P.2) 
  (h2 : g b c P.1 = P.2) 
  (h3 : (deriv (f a)) P.1 = (deriv (g b c)) P.1) :
  (∃ (k : ℝ), 
    (∀ x, f a x = 2 * x^3 - 8 * x) ∧ 
    (∀ x, g b c x = 4 * x^2 - 16) ∧
    (∀ x y, k * x - y - k * P.1 + P.2 = 0 ↔ y = (deriv (f a)) P.1 * (x - P.1) + P.2)) :=
sorry

end

end functions_and_tangent_line_l3422_342272


namespace point_transformation_l3422_342291

def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 7 ∧ y₂ = -4) → d - c = 3 := by
  sorry

end point_transformation_l3422_342291


namespace muffin_banana_price_ratio_l3422_342296

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ), m > 0 → b > 0 →
  (5 * m + 4 * b) * 3 = 3 * m + 20 * b →
  m / b = 2 / 3 := by
sorry

end muffin_banana_price_ratio_l3422_342296


namespace trent_total_travel_l3422_342230

/-- Represents the number of blocks Trent traveled -/
def trent_travel (blocks_to_bus_stop : ℕ) (blocks_on_bus : ℕ) : ℕ :=
  2 * (blocks_to_bus_stop + blocks_on_bus)

/-- Proves that Trent's total travel is 22 blocks given the problem conditions -/
theorem trent_total_travel :
  trent_travel 4 7 = 22 := by
  sorry

end trent_total_travel_l3422_342230


namespace eustace_milford_age_ratio_l3422_342234

theorem eustace_milford_age_ratio :
  ∀ (eustace_age milford_age : ℕ),
    eustace_age + 3 = 39 →
    milford_age + 3 = 21 →
    eustace_age / milford_age = 2 := by
  sorry

end eustace_milford_age_ratio_l3422_342234


namespace dans_cards_count_l3422_342265

/-- The number of Pokemon cards Dan gave to Sally -/
def cards_from_dan (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : ℕ :=
  total_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 Pokemon cards -/
theorem dans_cards_count : cards_from_dan 27 20 88 = 41 := by
  sorry

end dans_cards_count_l3422_342265


namespace marble_remainder_l3422_342275

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by
sorry

end marble_remainder_l3422_342275


namespace stratified_sampling_l3422_342208

/-- Represents the number of students in each grade and the sample size -/
structure SchoolSample where
  total : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_first : ℕ

/-- The conditions of the problem -/
def school_conditions (s : SchoolSample) : Prop :=
  s.total = 1290 ∧
  s.first_grade = 480 ∧
  s.second_grade = s.third_grade + 30 ∧
  s.total = s.first_grade + s.second_grade + s.third_grade ∧
  s.sample_first = 96

/-- The theorem to prove -/
theorem stratified_sampling (s : SchoolSample) 
  (h : school_conditions s) : 
  (s.sample_first * s.second_grade) / s.first_grade = 78 := by
  sorry


end stratified_sampling_l3422_342208


namespace geometric_arithmetic_sequence_l3422_342241

theorem geometric_arithmetic_sequence (a b c : ℝ) (q : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Ensure positive terms
  a > b → b > c →  -- Decreasing sequence
  b = a * q →  -- Geometric progression
  c = b * q →  -- Geometric progression
  2 * (2020 * b / 7) = 577 * a + c / 7 →  -- Arithmetic progression condition
  q = 1 / 2 := by sorry

end geometric_arithmetic_sequence_l3422_342241


namespace factor_x_squared_minus_64_l3422_342235

theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end factor_x_squared_minus_64_l3422_342235


namespace acetone_weight_approx_l3422_342201

/-- Atomic weight of Carbon in amu -/
def carbon_weight : Float := 12.01

/-- Atomic weight of Hydrogen in amu -/
def hydrogen_weight : Float := 1.008

/-- Atomic weight of Oxygen in amu -/
def oxygen_weight : Float := 16.00

/-- Number of Carbon atoms in Acetone -/
def carbon_count : Nat := 3

/-- Number of Hydrogen atoms in Acetone -/
def hydrogen_count : Nat := 6

/-- Number of Oxygen atoms in Acetone -/
def oxygen_count : Nat := 1

/-- Calculates the molecular weight of Acetone -/
def acetone_molecular_weight : Float :=
  carbon_weight * carbon_count.toFloat +
  hydrogen_weight * hydrogen_count.toFloat +
  oxygen_weight * oxygen_count.toFloat

/-- Theorem stating that the molecular weight of Acetone is approximately 58.08 amu -/
theorem acetone_weight_approx :
  (acetone_molecular_weight - 58.08).abs < 0.01 := by
  sorry

end acetone_weight_approx_l3422_342201


namespace complement_of_A_range_of_c_l3422_342203

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

-- Define set B
def B (c : ℝ) : Set ℝ := {x : ℝ | x > c}

-- Theorem for the complement of A
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for the range of c
theorem range_of_c :
  (∀ x : ℝ, x ∈ A ∨ x ∈ B c) → c ∈ Set.Iic (-2) := by sorry

end complement_of_A_range_of_c_l3422_342203


namespace min_value_of_sum_of_squares_l3422_342245

theorem min_value_of_sum_of_squares (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 160 := by
  sorry

end min_value_of_sum_of_squares_l3422_342245


namespace log_and_inverse_properties_l3422_342297

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function of log_a(x)
noncomputable def log_inverse (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Theorem statement
theorem log_and_inverse_properties (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  -- 1. Same monotonicity
  (∀ x y, x < y → log a x < log a y ↔ log_inverse a x < log_inverse a y) ∧
  -- 2. No intersection when a > 1
  (a > 1 → ∀ x, log a x ≠ log_inverse a x) ∧
  -- 3. Intersection point on y = x
  (∀ x, log a x = log_inverse a x → log a x = x) :=
by sorry

end log_and_inverse_properties_l3422_342297


namespace floor_difference_equals_five_l3422_342282

theorem floor_difference_equals_five (n : ℤ) : (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5) ↔ (n = 11) :=
sorry

end floor_difference_equals_five_l3422_342282


namespace ellipse_equation_l3422_342261

/-- Given an ellipse with foci F₁(0,-4) and F₂(0,4), and the shortest distance from a point on the ellipse to F₁ is 2, the equation of the ellipse is x²/20 + y²/36 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let f₁ : ℝ × ℝ := (0, -4)
  let f₂ : ℝ × ℝ := (0, 4)
  let shortest_distance : ℝ := 2
  x^2 / 20 + y^2 / 36 = 1 :=
by sorry

end ellipse_equation_l3422_342261


namespace melissa_driving_hours_l3422_342278

/-- Calculates the total driving hours in a year for a person who drives to town twice each month -/
def yearly_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * 12

theorem melissa_driving_hours :
  yearly_driving_hours 2 3 = 72 := by
sorry

end melissa_driving_hours_l3422_342278


namespace max_s_value_l3422_342249

/-- Given two regular polygons P₁ (r-gon) and P₂ (s-gon), where the interior angle of P₁ is 68/67 times
    the interior angle of P₂, this theorem states that the maximum possible value of s is 135. -/
theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) 
  (h_angle : (r - 2) * s * 68 = (s - 2) * r * 67) : s ≤ 135 ∧ ∃ r : ℕ, r ≥ 135 ∧ (135 - 2) * r * 67 = (r - 2) * 135 * 68 := by
  sorry

#check max_s_value

end max_s_value_l3422_342249


namespace smallest_x_is_correct_l3422_342255

/-- The smallest positive integer x such that 1680x is a perfect cube -/
def smallest_x : ℕ := 44100

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ m : ℤ, 1680 * y = m^3) ∧
  ∃ m : ℤ, 1680 * smallest_x = m^3 := by
  sorry

#eval smallest_x

end smallest_x_is_correct_l3422_342255


namespace adams_balls_l3422_342260

theorem adams_balls (red : ℕ) (blue : ℕ) (pink : ℕ) (orange : ℕ) : 
  red = 20 → 
  blue = 10 → 
  pink = 3 * orange → 
  orange = 5 → 
  red + blue + pink + orange = 50 := by
sorry

end adams_balls_l3422_342260


namespace shaded_region_area_l3422_342209

/-- Given a figure composed of 25 congruent squares, where the diagonal of a square 
    formed by 16 of these squares is 10 cm, the total area of the figure is 78.125 square cm. -/
theorem shaded_region_area (num_squares : ℕ) (diagonal : ℝ) (total_area : ℝ) : 
  num_squares = 25 → 
  diagonal = 10 → 
  total_area = 78.125 := by
  sorry

end shaded_region_area_l3422_342209


namespace largest_angle_is_70_l3422_342221

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_eq_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define the specific conditions of our triangle
def special_triangle (t : Triangle) : Prop :=
  ∃ (x : ℝ), 
    t.angle1 = x ∧ 
    t.angle2 = x + 20 ∧
    t.angle1 + t.angle2 = (4/3) * right_angle

-- Theorem statement
theorem largest_angle_is_70 (t : Triangle) (h : special_triangle t) : 
  max t.angle1 (max t.angle2 t.angle3) = 70 :=
sorry

end largest_angle_is_70_l3422_342221


namespace rectangle_area_l3422_342214

def length (x : ℝ) : ℝ := 5 * x + 3

def width (x : ℝ) : ℝ := x - 7

def area (x : ℝ) : ℝ := length x * width x

theorem rectangle_area (x : ℝ) : area x = 5 * x^2 - 32 * x - 21 := by
  sorry

end rectangle_area_l3422_342214


namespace adult_ticket_cost_l3422_342292

theorem adult_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) 
  (h1 : num_adults = 5) 
  (h2 : num_children = 2) 
  (h3 : concession_cost = 12) 
  (h4 : total_cost = 76) 
  (h5 : child_ticket_cost = 7) :
  ∃ (adult_ticket_cost : ℚ), 
    adult_ticket_cost = 10 ∧ 
    (num_adults : ℚ) * adult_ticket_cost + 
    (num_children : ℚ) * child_ticket_cost + 
    concession_cost = total_cost :=
by
  sorry

end adult_ticket_cost_l3422_342292


namespace inequality_proof_l3422_342223

theorem inequality_proof (x₃ x₄ : ℝ) (h1 : 1 < x₃) (h2 : x₃ < x₄) :
  x₃ * Real.exp x₄ > x₄ * Real.exp x₃ := by sorry

end inequality_proof_l3422_342223


namespace power_mod_eleven_l3422_342286

theorem power_mod_eleven : 7^79 ≡ 6 [ZMOD 11] := by sorry

end power_mod_eleven_l3422_342286


namespace triangle_inequality_condition_l3422_342254

theorem triangle_inequality_condition (k : ℕ) : 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  k ≥ 6 := by
sorry


end triangle_inequality_condition_l3422_342254


namespace subtract_from_zero_is_additive_inverse_l3422_342228

theorem subtract_from_zero_is_additive_inverse (a : ℚ) : 0 - a = -a := by sorry

end subtract_from_zero_is_additive_inverse_l3422_342228


namespace complement_A_union_B_a_lower_bound_l3422_342259

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x < 1 ∨ x > 2} := by sorry

-- Theorem for part II
theorem a_lower_bound (h : A ⊆ C a) : a ≥ 7 := by sorry

end complement_A_union_B_a_lower_bound_l3422_342259


namespace inequality_holds_l3422_342299

theorem inequality_holds (r s : ℝ) : 
  r ≥ 0 → s > 0 → 
  (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) ≤ 3 * r^2 * s ↔ 
  r ≥ (2 + 2 * Real.sqrt 13) / 3 := by
sorry

end inequality_holds_l3422_342299


namespace smallest_five_digit_mod_11_l3422_342248

theorem smallest_five_digit_mod_11 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 11 = 9 → n ≥ 10000 :=
by sorry

end smallest_five_digit_mod_11_l3422_342248


namespace complex_roots_on_circle_l3422_342212

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z - 2)^4 = 16 * z^4 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
sorry

end complex_roots_on_circle_l3422_342212


namespace number_with_given_division_l3422_342202

theorem number_with_given_division : ∃ n : ℕ, n = 100 ∧ n / 11 = 9 ∧ n % 11 = 1 := by
  sorry

end number_with_given_division_l3422_342202


namespace seven_twentyfour_twentyfive_pythagorean_triple_l3422_342242

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 :=
by
  sorry

end seven_twentyfour_twentyfive_pythagorean_triple_l3422_342242


namespace tangent_line_b_value_l3422_342206

-- Define the curve and line
def curve (x b c : ℝ) : ℝ := x^3 + b*x^2 + c
def line (x k : ℝ) : ℝ := k*x + 1

-- Define the derivative of the curve
def curve_derivative (x b : ℝ) : ℝ := 3*x^2 + 2*b*x

theorem tangent_line_b_value :
  ∀ (k b c : ℝ),
  -- The line passes through the point (1, 2)
  (line 1 k = 2) →
  -- The curve passes through the point (1, 2)
  (curve 1 b c = 2) →
  -- The slope of the line equals the derivative of the curve at x = 1
  (k = curve_derivative 1 b) →
  -- The value of b is -1
  (b = -1) := by sorry

end tangent_line_b_value_l3422_342206


namespace divisibility_by_five_l3422_342290

theorem divisibility_by_five (B : ℕ) : 
  B < 10 → (947 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end divisibility_by_five_l3422_342290


namespace coin_worth_l3422_342280

def total_coins : ℕ := 20
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def swap_difference : ℚ := 70 / 100

theorem coin_worth (n : ℕ) (h1 : n ≤ total_coins) :
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value + swap_difference = 
  (n : ℚ) * dime_value + (total_coins - n : ℚ) * nickel_value →
  (n : ℚ) * nickel_value + (total_coins - n : ℚ) * dime_value = 115 / 100 := by
  sorry

end coin_worth_l3422_342280


namespace sum_of_solutions_is_zero_l3422_342251

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) (y : ℝ) : 
  y = 5 → 
  x₁^2 + y^2 = 169 → 
  x₂^2 + y^2 = 169 → 
  x₁ + x₂ = 0 := by
sorry

end sum_of_solutions_is_zero_l3422_342251


namespace count_descending_digit_numbers_l3422_342238

/-- The number of natural numbers with 2 or more digits where each subsequent digit is less than the previous one -/
def descending_digit_numbers : ℕ :=
  (Finset.range 9).sum (fun k => Nat.choose 10 (k + 2))

/-- Theorem stating that the number of natural numbers with 2 or more digits 
    where each subsequent digit is less than the previous one is 1013 -/
theorem count_descending_digit_numbers : descending_digit_numbers = 1013 := by
  sorry

end count_descending_digit_numbers_l3422_342238


namespace secret_spread_theorem_l3422_342224

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day of the week given a number of days after Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n = 3280 ∧ day_of_week n = "Monday" :=
by
  sorry

end secret_spread_theorem_l3422_342224


namespace cube_root_three_equation_l3422_342222

theorem cube_root_three_equation (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
sorry

end cube_root_three_equation_l3422_342222


namespace final_digit_mod_seven_l3422_342216

/-- Represents the allowed operations on the number --/
inductive Operation
  | increaseDecrease : Operation
  | subtractAdd : Operation
  | decreaseBySeven : Operation

/-- The initial number as a list of digits --/
def initialNumber : List Nat := List.replicate 100 8

/-- A function that applies an operation to a list of digits --/
def applyOperation (digits : List Nat) (op : Operation) : List Nat :=
  sorry

/-- A function that removes leading zeros from a list of digits --/
def removeLeadingZeros (digits : List Nat) : List Nat :=
  sorry

/-- A function that applies operations until a single digit remains --/
def applyOperationsUntilSingleDigit (digits : List Nat) : Nat :=
  sorry

/-- Theorem stating that the final single digit is equivalent to 3 modulo 7 --/
theorem final_digit_mod_seven (ops : List Operation) :
  (applyOperationsUntilSingleDigit initialNumber) % 7 = 3 :=
sorry

end final_digit_mod_seven_l3422_342216


namespace fifth_scroll_age_l3422_342233

def scroll_age (n : ℕ) : ℕ → ℕ
  | 0 => 4080
  | (m + 1) => scroll_age n m + (scroll_age n m) / 2

theorem fifth_scroll_age : scroll_age 5 4 = 20655 := by
  sorry

end fifth_scroll_age_l3422_342233


namespace oranges_from_first_tree_l3422_342285

/-- Represents the number of oranges picked from each tree -/
structure OrangesPicked where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The total number of oranges picked is the sum of oranges from all three trees -/
def total_oranges (op : OrangesPicked) : ℕ := op.first + op.second + op.third

/-- Theorem: Given the total oranges and the number from the second and third trees, 
    we can determine the number of oranges from the first tree -/
theorem oranges_from_first_tree (op : OrangesPicked) 
  (h1 : total_oranges op = 260)
  (h2 : op.second = 60)
  (h3 : op.third = 120) :
  op.first = 80 := by
  sorry

end oranges_from_first_tree_l3422_342285


namespace triangle_special_progression_l3422_342219

theorem triangle_special_progression (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Arithmetic progression of sides
  2 * b = a + c →
  -- Geometric progression of sines
  Real.sin B ^ 2 = Real.sin A * Real.sin C →
  -- Conclusion
  B = π / 3 := by
sorry

end triangle_special_progression_l3422_342219


namespace rice_price_fall_l3422_342271

theorem rice_price_fall (original_price : ℝ) (original_quantity : ℝ) : 
  original_price > 0 →
  original_quantity > 0 →
  let new_price := 0.8 * original_price
  let new_quantity := 50
  original_price * original_quantity = new_price * new_quantity →
  original_quantity = 40 := by
sorry

end rice_price_fall_l3422_342271


namespace complex_fraction_equals_negative_two_l3422_342220

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_negative_two :
  (1 + i)^3 / (1 - i) = -2 := by sorry

end complex_fraction_equals_negative_two_l3422_342220


namespace intersection_distance_l3422_342276

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection1 : ℝ × ℝ := (1, 1)
def intersection2 : ℝ × ℝ := (0, 0)

-- State the theorem
theorem intersection_distance :
  curve1 (intersection1.1) (intersection1.2) ∧
  curve2 (intersection1.1) (intersection1.2) ∧
  curve1 (intersection2.1) (intersection2.2) ∧
  curve2 (intersection2.1) (intersection2.2) →
  Real.sqrt ((intersection1.1 - intersection2.1)^2 + (intersection1.2 - intersection2.2)^2) = Real.sqrt 2 :=
by sorry

end intersection_distance_l3422_342276


namespace large_cube_edge_approx_l3422_342213

/-- The edge length of a smaller cube in centimeters -/
def small_cube_edge : ℝ := 20

/-- The approximate number of smaller cubes that fit in the larger cubical box -/
def num_small_cubes : ℝ := 125

/-- The approximate edge length of the larger cubical box in centimeters -/
def large_cube_edge : ℝ := 100

/-- Theorem stating that the edge length of the larger cubical box is approximately 100 cm -/
theorem large_cube_edge_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |large_cube_edge ^ 3 - num_small_cubes * small_cube_edge ^ 3| < ε * (num_small_cubes * small_cube_edge ^ 3) :=
sorry

end large_cube_edge_approx_l3422_342213


namespace binomial_arithmetic_sequence_implies_seven_l3422_342293

def factorial (r : ℕ) : ℕ := Nat.factorial r

def binomial_coefficient (j k : ℕ) : ℕ :=
  if k ≤ j then
    factorial j / (factorial k * factorial (j - k))
  else
    0

theorem binomial_arithmetic_sequence_implies_seven (n : ℕ) 
  (h1 : n > 3)
  (h2 : ∃ d : ℕ, binomial_coefficient n 2 - binomial_coefficient n 1 = d ∧
                 binomial_coefficient n 3 - binomial_coefficient n 2 = d) :
  n = 7 := by
  sorry

end binomial_arithmetic_sequence_implies_seven_l3422_342293


namespace cost_of_apple_and_watermelon_l3422_342294

/-- Represents the price of fruits in yuan per kilogram -/
structure FruitPrices where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ

/-- Represents a purchase of fruits -/
structure Purchase where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ
  total : ℝ

def xiaoming_purchase : Purchase :=
  { apple := 1, watermelon := 4, orange := 2, total := 27.2 }

def xiaohui_purchase : Purchase :=
  { apple := 2, watermelon := 6, orange := 2, total := 32.4 }

theorem cost_of_apple_and_watermelon (prices : FruitPrices) :
  xiaoming_purchase.apple * prices.apple +
  xiaoming_purchase.watermelon * prices.watermelon +
  xiaoming_purchase.orange * prices.orange = xiaoming_purchase.total ∧
  xiaohui_purchase.apple * prices.apple +
  xiaohui_purchase.watermelon * prices.watermelon +
  xiaohui_purchase.orange * prices.orange = xiaohui_purchase.total →
  prices.apple + 2 * prices.watermelon = 5.2 := by
  sorry

end cost_of_apple_and_watermelon_l3422_342294


namespace average_books_read_rounded_l3422_342250

/-- Represents the number of books read by each category of members -/
def books_read : List Nat := [1, 2, 3, 4, 5]

/-- Represents the number of members in each category -/
def members : List Nat := [3, 4, 1, 6, 2]

/-- Calculates the total number of books read -/
def total_books : Nat := (List.zip books_read members).map (fun (b, m) => b * m) |>.sum

/-- Calculates the total number of members -/
def total_members : Nat := members.sum

/-- Calculates the average number of books read per member -/
def average : Rat := total_books / total_members

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : Rat) : Int :=
  ⌊x + 1/2⌋

/-- Main theorem: The average number of books read, rounded to the nearest whole number, is 3 -/
theorem average_books_read_rounded : round_to_nearest average = 3 := by
  sorry

end average_books_read_rounded_l3422_342250


namespace physical_examination_count_l3422_342243

theorem physical_examination_count (boys girls examined : ℕ) 
  (h1 : boys = 121)
  (h2 : girls = 83)
  (h3 : examined = 150) :
  boys + girls - examined = 54 := by
  sorry

end physical_examination_count_l3422_342243


namespace function_max_min_sum_l3422_342298

/-- Given a function f and a positive real number t, 
    this theorem states that if the sum of the maximum and minimum values of f is 4, 
    then t must equal 2. -/
theorem function_max_min_sum (t : ℝ) (h1 : t > 0) : 
  let f : ℝ → ℝ := λ x ↦ (t*x^2 + 2*x + t^2 + Real.sin x) / (x^2 + t)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, f x ≥ N) ∧ (M + N = 4) → t = 2 := by
  sorry

end function_max_min_sum_l3422_342298


namespace fraction_unchanged_l3422_342267

theorem fraction_unchanged (x y : ℝ) (h : x ≠ y) :
  (3 * x) / (x - y) = (3 * (2 * x)) / ((2 * x) - (2 * y)) :=
by sorry

end fraction_unchanged_l3422_342267


namespace carls_weight_l3422_342237

/-- Given the weights of Al, Ben, Carl, and Ed, prove Carl's weight -/
theorem carls_weight (Al Ben Carl Ed : ℕ) 
  (h1 : Al = Ben + 25)
  (h2 : Ben = Carl - 16)
  (h3 : Ed = 146)
  (h4 : Al = Ed + 38) :
  Carl = 175 := by
  sorry

end carls_weight_l3422_342237


namespace system_solutions_correct_l3422_342270

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, x - 3 * y = -10 ∧ x + y = 6 ∧ x = 2 ∧ y = 4) ∧
  -- System 2
  (∃ x y : ℝ, x / 2 - (y - 1) / 3 = 1 ∧ 4 * x - y = 8 ∧ x = 12 / 5 ∧ y = 8 / 5) :=
by
  sorry


end system_solutions_correct_l3422_342270


namespace simplify_fraction_l3422_342227

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by sorry

end simplify_fraction_l3422_342227


namespace sufficient_condition_for_square_inequality_l3422_342287

theorem sufficient_condition_for_square_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) →
  (a ≥ 5 → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0)) ∧
  ¬(∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 → a ≥ 5) :=
by sorry

end sufficient_condition_for_square_inequality_l3422_342287


namespace min_distance_circle_to_line_l3422_342266

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line x - y = 2 is √2 - 1 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - y = 2}
  (∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∀ (r : ℝ × ℝ), r ∈ line →
        dist p r ≥ Real.sqrt 2 - 1)) ∧
  (∃ (p : ℝ × ℝ) (r : ℝ × ℝ), p ∈ circle ∧ r ∈ line ∧ dist p r = Real.sqrt 2 - 1) :=
by sorry

end min_distance_circle_to_line_l3422_342266


namespace ab_minus_c_equals_six_l3422_342232

theorem ab_minus_c_equals_six (a b c : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c^2 = a*b + b - 9) : 
  a*b - c = 6 := by
sorry

end ab_minus_c_equals_six_l3422_342232


namespace remainder_of_n_squared_plus_4n_plus_10_l3422_342252

theorem remainder_of_n_squared_plus_4n_plus_10 (n : ℤ) (a : ℤ) (h : n = 100 * a - 2) :
  (n^2 + 4*n + 10) % 100 = 6 := by
sorry

end remainder_of_n_squared_plus_4n_plus_10_l3422_342252


namespace quadratic_completion_l3422_342205

theorem quadratic_completion (x : ℝ) : x^2 + 16*x + 72 = (x + 8)^2 + 8 := by
  sorry

end quadratic_completion_l3422_342205


namespace product_divisors_24_power_5_l3422_342210

/-- The product of divisors function -/
def prod_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem product_divisors_24_power_5 (n : ℕ+) :
  prod_divisors n = (24 : ℕ+) ^ 240 → n = (24 : ℕ+) ^ 5 := by
  sorry

end product_divisors_24_power_5_l3422_342210


namespace donald_oranges_l3422_342264

def final_oranges (initial found given_away : ℕ) : ℕ :=
  initial + found - given_away

theorem donald_oranges : 
  final_oranges 4 5 3 = 6 := by sorry

end donald_oranges_l3422_342264


namespace arithmetic_sequences_common_terms_l3422_342246

/-- The first arithmetic sequence -/
def seq1 (n : ℕ) : ℕ := 2 + 3 * n

/-- The second arithmetic sequence -/
def seq2 (n : ℕ) : ℕ := 4 + 5 * n

/-- The last term of the first sequence -/
def last1 : ℕ := 2015

/-- The last term of the second sequence -/
def last2 : ℕ := 2014

/-- The number of common terms between the two sequences -/
def commonTerms : ℕ := 134

theorem arithmetic_sequences_common_terms :
  (∃ (s : Finset ℕ), s.card = commonTerms ∧
    (∀ x ∈ s, ∃ n m : ℕ, seq1 n = x ∧ seq2 m = x ∧
      seq1 n ≤ last1 ∧ seq2 m ≤ last2)) :=
sorry

end arithmetic_sequences_common_terms_l3422_342246


namespace quadratic_inequality_problem_l3422_342274

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | 1 < x ∧ x < 3}

-- Define the solution set A
def A (a c : ℝ) := {x : ℝ | a * x^2 + 2*x + 4*c > 0}

-- Define the solution set B
def B (a c m : ℝ) := {x : ℝ | 3*a*x + c*m < 0}

-- State the theorem
theorem quadratic_inequality_problem 
  (h1 : ∀ x, x ∈ S ↔ f a c x > 0)
  (h2 : A a c ⊆ B a c m) :
  a = -1/4 ∧ c = -3/4 ∧ m ≥ -2 := by sorry

end quadratic_inequality_problem_l3422_342274


namespace vector_operation_l3422_342284

/-- Given vectors a and b in ℝ², prove that 2b - a equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  (2 : ℝ) • b - a = (-3, -4) := by sorry

end vector_operation_l3422_342284


namespace x_value_l3422_342277

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^3 + 2*y^2 - 2) / (y^3 + 2*y^2 - 3)) :
  x = (y^3 + 2*y^2 - 2) / 2 := by
sorry

end x_value_l3422_342277


namespace sum_of_digits_2023_base7_l3422_342256

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_2023_base7 :
  sumDigits (toBase7 2023) = 13 := by
  sorry

end sum_of_digits_2023_base7_l3422_342256


namespace geometric_relations_l3422_342273

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => parallel_plane
local infix:50 " ⊥ₚ " => perpendicular_plane
local infix:50 " ∥ₗₚ " => line_parallel_plane
local infix:50 " ⊥ₗₚ " => line_perpendicular_plane

-- Theorem statement
theorem geometric_relations (m n : Line) (α β : Plane) :
  (((m ⊥ₗₚ α) ∧ (n ⊥ₗₚ β) ∧ (α ⊥ₚ β)) → (m ⊥ n)) ∧
  (((m ⊥ₗₚ α) ∧ (n ∥ₗₚ β) ∧ (α ∥ₚ β)) → (m ⊥ n)) :=
sorry

end geometric_relations_l3422_342273


namespace sets_problem_l3422_342226

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 5 ≤ x ∧ x < 8}) ∧
  (C a ∩ A = C a → a ≤ -1) :=
by sorry

end sets_problem_l3422_342226


namespace quadrilateral_offset_l3422_342268

/-- Given a quadrilateral with one diagonal of length d, two offsets h1 and h2,
    and area A, this theorem states that if d = 30, h2 = 6, and A = 240,
    then h1 = 10. -/
theorem quadrilateral_offset (d h1 h2 A : ℝ) :
  d = 30 → h2 = 6 → A = 240 → A = (1/2) * d * (h1 + h2) → h1 = 10 := by
  sorry

#check quadrilateral_offset

end quadrilateral_offset_l3422_342268


namespace pentagon_perimeter_l3422_342269

/-- The perimeter of a pentagon with side lengths 2, √8, √18, √32, and √62 is 2 + 9√2 + √62 -/
theorem pentagon_perimeter : 
  let side1 : ℝ := 2
  let side2 : ℝ := Real.sqrt 8
  let side3 : ℝ := Real.sqrt 18
  let side4 : ℝ := Real.sqrt 32
  let side5 : ℝ := Real.sqrt 62
  side1 + side2 + side3 + side4 + side5 = 2 + 9 * Real.sqrt 2 + Real.sqrt 62 := by
  sorry


end pentagon_perimeter_l3422_342269


namespace pumps_to_fill_tires_l3422_342207

/-- Represents the capacity of a single tire in cubic inches -/
def tireCapacity : ℝ := 500

/-- Represents the amount of air injected per pump in cubic inches -/
def airPerPump : ℝ := 50

/-- Calculates the total air needed to fill all tires -/
def totalAirNeeded : ℝ :=
  2 * tireCapacity +  -- Two flat tires
  0.6 * tireCapacity +  -- Tire that's 40% full needs 60% more
  0.3 * tireCapacity  -- Tire that's 70% full needs 30% more

/-- Theorem: The number of pumps required to fill all tires is 29 -/
theorem pumps_to_fill_tires : 
  ⌈totalAirNeeded / airPerPump⌉ = 29 := by sorry

end pumps_to_fill_tires_l3422_342207


namespace distance_after_three_minutes_l3422_342204

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

/-- Theorem: The distance between two vehicles with speeds 65 km/h and 85 km/h after 3 minutes is 1 km -/
theorem distance_after_three_minutes :
  let v1 : ℝ := 65  -- Speed of the truck in km/h
  let v2 : ℝ := 85  -- Speed of the car in km/h
  let t : ℝ := 3 / 60  -- 3 minutes converted to hours
  distance_between_vehicles v1 v2 t = 1 := by
  sorry


end distance_after_three_minutes_l3422_342204


namespace solve_system_for_x_l3422_342244

theorem solve_system_for_x (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y + z = 8) 
  (eq2 : x + 3 * y - 2 * z = 2) : 
  x = 58 / 21 := by
sorry

end solve_system_for_x_l3422_342244


namespace prob_same_face_is_37_64_l3422_342217

/-- A cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

/-- A random vertex of a cube -/
def random_vertex (C : Cube) : Fin 8 := sorry

/-- The probability that three random vertices lie on the same face of a cube -/
def prob_same_face (C : Cube) : ℚ :=
  let P := random_vertex C
  let Q := random_vertex C
  let R := random_vertex C
  sorry

/-- Theorem: The probability that three random vertices of a cube lie on the same face is 37/64 -/
theorem prob_same_face_is_37_64 (C : Cube) : prob_same_face C = 37 / 64 := by
  sorry

end prob_same_face_is_37_64_l3422_342217


namespace ending_number_is_67_l3422_342247

-- Define the sum of first n odd integers
def sum_odd_integers (n : ℕ) : ℕ := n^2

-- Define the sum of odd integers from a to b inclusive
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_odd_integers ((b - a) / 2 + 1) - sum_odd_integers ((a - 1) / 2)

-- The main theorem
theorem ending_number_is_67 :
  ∃ x : ℕ, x ≥ 11 ∧ sum_odd_range 11 x = 416 ∧ x = 67 :=
sorry

end ending_number_is_67_l3422_342247


namespace baby_sea_turtles_on_sand_l3422_342239

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (on_sand : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  on_sand = total - (swept_fraction * total).num → 
  on_sand = 28 := by
sorry

end baby_sea_turtles_on_sand_l3422_342239


namespace quadratic_equation_roots_range_l3422_342218

theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 - 2 * x + 1 = 0 ∧ (k - 1) * y^2 - 2 * y + 1 = 0) →
  k ≤ 2 ∧ k ≠ 1 :=
by sorry

end quadratic_equation_roots_range_l3422_342218


namespace equation_solution_l3422_342229

theorem equation_solution (a b : ℝ) (h : a - b = 0) : 
  ∃! x : ℝ, a * x + b = 0 ∧ x = -1 :=
by sorry

end equation_solution_l3422_342229


namespace fraction_equivalence_l3422_342253

theorem fraction_equivalence : 
  let original := 8 / 9
  let target := 4 / 5
  let subtracted := 4
  (8 - subtracted) / (9 - subtracted) = target := by sorry

end fraction_equivalence_l3422_342253
