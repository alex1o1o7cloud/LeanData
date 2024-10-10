import Mathlib

namespace range_of_a_l2832_283205

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ∈ {x : ℝ | x ≥ 5} := by
  sorry

end range_of_a_l2832_283205


namespace initial_mixture_volume_l2832_283267

/-- Proves that given a mixture with 10% water content, if 5 liters of water are added
    to make the new mixture contain 20% water, then the initial volume of the mixture was 40 liters. -/
theorem initial_mixture_volume
  (initial_water_percentage : Real)
  (added_water : Real)
  (final_water_percentage : Real)
  (h1 : initial_water_percentage = 0.10)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 0.20)
  : ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water
      = (initial_volume + added_water) * final_water_percentage
    ∧ initial_volume = 40 := by
  sorry

end initial_mixture_volume_l2832_283267


namespace composite_surface_area_is_39_l2832_283248

/-- The surface area of a composite object formed by three cylinders -/
def composite_surface_area (π : ℝ) (h : ℝ) (r₁ r₂ r₃ : ℝ) : ℝ :=
  (2 * π * r₁ * h + π * r₁^2) +
  (2 * π * r₂ * h + π * r₂^2) +
  (2 * π * r₃ * h + π * r₃^2) +
  π * r₁^2 + π * r₂^2 + π * r₃^2

/-- The surface area of the composite object is 39 square meters -/
theorem composite_surface_area_is_39 :
  composite_surface_area 3 1 1.5 1 0.5 = 39 := by
  sorry

end composite_surface_area_is_39_l2832_283248


namespace number_of_possible_sets_l2832_283224

theorem number_of_possible_sets (A : Set ℤ) : 
  (A ∪ {-1, 1} = {-1, 0, 1}) → 
  (∃ (S : Finset (Set ℤ)), (∀ X ∈ S, X ∪ {-1, 1} = {-1, 0, 1}) ∧ S.card = 4 ∧ 
    ∀ Y, Y ∪ {-1, 1} = {-1, 0, 1} → Y ∈ S) := by
  sorry

end number_of_possible_sets_l2832_283224


namespace eeyore_triangle_problem_l2832_283227

/-- A type representing a stick with a length -/
structure Stick :=
  (length : ℝ)

/-- A function to check if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- A function to split six sticks into two sets of three, with the three shortest in one set -/
def splitSticks (sticks : Fin 6 → Stick) : (Fin 3 → Stick) × (Fin 3 → Stick) :=
  sorry

theorem eeyore_triangle_problem :
  ∃ (sticks : Fin 6 → Stick),
    (∃ (t1 t2 t3 t4 t5 t6 : Fin 6), canFormTriangle (sticks t1) (sticks t2) (sticks t3) ∧
                                    canFormTriangle (sticks t4) (sticks t5) (sticks t6)) ∧
    let (yellow, green) := splitSticks sticks
    ¬(canFormTriangle (yellow 0) (yellow 1) (yellow 2) ∧
      canFormTriangle (green 0) (green 1) (green 2)) :=
  sorry

end eeyore_triangle_problem_l2832_283227


namespace xy_range_and_min_x_plus_2y_l2832_283252

theorem xy_range_and_min_x_plus_2y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x*y = 3) : 
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b = 3 → x + 2*y ≤ a + 2*b) ∧
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + d + c*d = 3 ∧ c + 2*d = 4*Real.sqrt 2 - 3) :=
by sorry

end xy_range_and_min_x_plus_2y_l2832_283252


namespace curve_crosses_itself_l2832_283231

/-- The x-coordinate of a point on the curve -/
def x (t k : ℝ) : ℝ := t^2 + k

/-- The y-coordinate of a point on the curve -/
def y (t k : ℝ) : ℝ := t^3 - k*t + 5

/-- Theorem stating that the curve crosses itself at (18,5) when k = 9 -/
theorem curve_crosses_itself : 
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    x t₁ 9 = x t₂ 9 ∧ 
    y t₁ 9 = y t₂ 9 ∧
    x t₁ 9 = 18 ∧ 
    y t₁ 9 = 5 :=
sorry

end curve_crosses_itself_l2832_283231


namespace harmonic_quadratic_radical_simplification_l2832_283247

theorem harmonic_quadratic_radical_simplification :
  ∃ (x y : ℕ+), (x + y : ℝ) = 11 ∧ (x * y : ℝ) = 28 →
  Real.sqrt (11 + 2 * Real.sqrt 28) = 2 + Real.sqrt 7 := by
  sorry

end harmonic_quadratic_radical_simplification_l2832_283247


namespace partner_calculation_l2832_283207

theorem partner_calculation (x : ℝ) : 4 * (3 * (x + 2) - 2) = 4 * (3 * x + 4) := by
  sorry

#check partner_calculation

end partner_calculation_l2832_283207


namespace stamp_selection_l2832_283221

theorem stamp_selection (n k : ℕ) (stamps : Finset ℕ) : 
  0 < n → 
  stamps.card = k → 
  n ≤ stamps.sum id → 
  stamps.sum id < 2 * k → 
  ∃ s : Finset ℕ, s ⊆ stamps ∧ s.sum id = n := by
  sorry

#check stamp_selection

end stamp_selection_l2832_283221


namespace total_classes_is_nine_l2832_283271

/-- The number of classes taught by Eduardo and Frankie -/
def total_classes (eduardo_classes : ℕ) (frankie_multiplier : ℕ) : ℕ :=
  eduardo_classes + eduardo_classes * frankie_multiplier

/-- Theorem stating that the total number of classes taught by Eduardo and Frankie is 9 -/
theorem total_classes_is_nine :
  total_classes 3 2 = 9 := by
  sorry

end total_classes_is_nine_l2832_283271


namespace boat_speed_current_l2832_283297

/-- Proves that given a boat with a constant speed of 16 mph relative to water,
    making an upstream trip in 20 minutes and a downstream trip in 15 minutes,
    the speed of the current is 16/7 mph. -/
theorem boat_speed_current (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  boat_speed = 16 ∧ upstream_time = 20 / 60 ∧ downstream_time = 15 / 60 →
  ∃ current_speed : ℝ,
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 :=
by sorry

end boat_speed_current_l2832_283297


namespace union_of_A_and_B_l2832_283235

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3} := by sorry

end union_of_A_and_B_l2832_283235


namespace square_sum_equals_ten_l2832_283279

theorem square_sum_equals_ten (a b : ℝ) 
  (h1 : a + 3 = (b - 1)^2) 
  (h2 : b + 3 = (a - 1)^2) 
  (h3 : a ≠ b) : 
  a^2 + b^2 = 10 := by
sorry

end square_sum_equals_ten_l2832_283279


namespace exactly_two_points_l2832_283238

/-- Given two points A and B in a plane that are 12 units apart, this function
    returns the number of points C such that triangle ABC has a perimeter of 36 units,
    an area of 72 square units, and is isosceles. -/
def count_valid_points (A B : ℝ × ℝ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly two points C satisfying the conditions. -/
theorem exactly_two_points (A B : ℝ × ℝ) 
    (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12) : 
    count_valid_points A B = 2 :=
  sorry

end exactly_two_points_l2832_283238


namespace A_greater_than_B_l2832_283217

theorem A_greater_than_B (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) > a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end A_greater_than_B_l2832_283217


namespace monkey_swinging_speed_l2832_283230

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem: The monkey's swinging speed is 10 feet per second --/
theorem monkey_swinging_speed 
  (running_speed : ℝ) 
  (running_time : ℝ) 
  (swinging_time : ℝ) 
  (total_distance : ℝ)
  (h1 : running_speed = 15)
  (h2 : running_time = 5)
  (h3 : swinging_time = 10)
  (h4 : total_distance = 175)
  (h5 : totalDistance 
    { speed := running_speed, time := running_time } 
    { speed := (total_distance - running_speed * running_time) / swinging_time, time := swinging_time } = total_distance) :
  (total_distance - running_speed * running_time) / swinging_time = 10 := by
  sorry

#check monkey_swinging_speed

end monkey_swinging_speed_l2832_283230


namespace flight_distance_difference_l2832_283276

def beka_flights : List Nat := [425, 320, 387]
def jackson_flights : List Nat := [250, 170, 353, 201]

theorem flight_distance_difference :
  (List.sum beka_flights) - (List.sum jackson_flights) = 158 := by
  sorry

end flight_distance_difference_l2832_283276


namespace difference_of_squares_312_308_l2832_283296

theorem difference_of_squares_312_308 : 312^2 - 308^2 = 2480 := by
  sorry

end difference_of_squares_312_308_l2832_283296


namespace complement_of_union_M_N_l2832_283264

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end complement_of_union_M_N_l2832_283264


namespace circle_radii_order_l2832_283270

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 10 →
  2 * Real.pi * rB = 10 * Real.pi →
  Real.pi * rC^2 = 25 * Real.pi →
  rA ≤ rB ∧ rB ≤ rC := by
  sorry

end circle_radii_order_l2832_283270


namespace basketball_win_rate_l2832_283282

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (games_won : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  games_won ≤ first_part_games →
  target_percentage = 3 / 4 →
  ∃ (x : ℕ), x ≤ remaining_games ∧ 
    (games_won + x : ℚ) / total_games = target_percentage ∧
    x = 38 :=
by
  sorry

#check basketball_win_rate 130 80 60 50 (3/4)

end basketball_win_rate_l2832_283282


namespace oblique_triangular_prism_volume_l2832_283263

/-- The volume of an oblique triangular prism with specific properties -/
theorem oblique_triangular_prism_volume (a : ℝ) (h : a > 0) :
  let base_area := (a^2 * Real.sqrt 3) / 4
  let height := a * Real.sqrt 3 / 2
  base_area * height = (3 * a^3) / 8 := by
  sorry

#check oblique_triangular_prism_volume

end oblique_triangular_prism_volume_l2832_283263


namespace least_m_for_x_bound_l2832_283259

def x : ℕ → ℚ
  | 0 => 3
  | n + 1 => (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_for_x_bound :
  ∃ m : ℕ, m = 33 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k < m, x k > 3 + 1 / 2^10 :=
sorry

end least_m_for_x_bound_l2832_283259


namespace ellipse_foci_distance_l2832_283240

/-- Given three points that represent three of the four endpoints of the axes of an ellipse -/
def point1 : ℝ × ℝ := (-2, 4)
def point2 : ℝ × ℝ := (3, -2)
def point3 : ℝ × ℝ := (8, 4)

/-- The theorem stating that the distance between the foci of the ellipse is 2√11 -/
theorem ellipse_foci_distance :
  ∃ (a b : ℝ) (center : ℝ × ℝ),
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    (center.1 - a = point1.1 ∨ center.1 - a = point2.1 ∨ center.1 - a = point3.1) ∧
    (center.1 + a = point1.1 ∨ center.1 + a = point2.1 ∨ center.1 + a = point3.1) ∧
    (center.2 - b = point1.2 ∨ center.2 - b = point2.2 ∨ center.2 - b = point3.2) ∧
    (center.2 + b = point1.2 ∨ center.2 + b = point2.2 ∨ center.2 + b = point3.2) ∧
    2 * Real.sqrt (max a b ^ 2 - min a b ^ 2) = 2 * Real.sqrt 11 :=
by sorry

end ellipse_foci_distance_l2832_283240


namespace function_derivative_positive_l2832_283255

/-- Given a function y = 2mx^2 + (1-4m)x + 2m - 1, prove that when m = -1 and x < 5/4, 
    the derivative of y with respect to x is positive. -/
theorem function_derivative_positive (x : ℝ) (h : x < 5/4) : 
  let m : ℝ := -1
  let y : ℝ → ℝ := λ x => 2*m*x^2 + (1-4*m)*x + 2*m - 1
  (deriv y) x > 0 := by sorry

end function_derivative_positive_l2832_283255


namespace no_real_roots_l2832_283237

theorem no_real_roots (a : ℝ) : (∀ x : ℝ, |x| ≠ a * x + 1) ↔ a ≥ 1 := by
  sorry

end no_real_roots_l2832_283237


namespace hyperbola_eccentricity_l2832_283226

/-- Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a = b) →  -- Perpendicular asymptotes condition
  (2 * (a^2 + b^2).sqrt = 8) →  -- Focal length condition
  ((a^2 + b^2).sqrt / a = Real.sqrt 2) := by
  sorry

#check hyperbola_eccentricity

end hyperbola_eccentricity_l2832_283226


namespace call_duration_is_60_minutes_l2832_283250

/-- Represents the duration of a single customer call in minutes. -/
def call_duration (cost_per_minute : ℚ) (monthly_bill : ℚ) (customers_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  (monthly_bill / cost_per_minute) / (customers_per_week * weeks_per_month)

/-- Theorem stating that under the given conditions, each call lasts 60 minutes. -/
theorem call_duration_is_60_minutes :
  call_duration (5 / 100) 600 50 4 = 60 := by
  sorry

end call_duration_is_60_minutes_l2832_283250


namespace linear_equation_passes_through_points_l2832_283208

/-- The linear equation passing through points A(1, 2) and B(3, 4) -/
def linear_equation (x y : ℝ) : Prop := y = x + 1

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 4)

/-- Theorem: The linear equation passes through points A and B -/
theorem linear_equation_passes_through_points :
  linear_equation point_A.1 point_A.2 ∧ linear_equation point_B.1 point_B.2 := by
  sorry

end linear_equation_passes_through_points_l2832_283208


namespace octagonal_pyramid_cross_section_distance_l2832_283254

-- Define the pyramid and cross sections
structure OctagonalPyramid where
  crossSection1Area : ℝ
  crossSection2Area : ℝ
  planeDistance : ℝ

-- Define the theorem
theorem octagonal_pyramid_cross_section_distance
  (pyramid : OctagonalPyramid)
  (h1 : pyramid.crossSection1Area = 324 * Real.sqrt 2)
  (h2 : pyramid.crossSection2Area = 648 * Real.sqrt 2)
  (h3 : pyramid.planeDistance = 12)
  : ∃ (distance : ℝ), distance = 24 + 12 * Real.sqrt 2 ∧
    distance = (pyramid.planeDistance) / (1 - Real.sqrt 2 / 2) :=
by sorry

end octagonal_pyramid_cross_section_distance_l2832_283254


namespace rs_length_l2832_283200

/-- Triangle ABC with altitude CH, inscribed circles tangent points R and S --/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  M : ℝ × ℝ
  -- CH is altitude
  altitude : (C.1 - H.1) * (A.1 - B.1) + (C.2 - H.2) * (A.2 - B.2) = 0
  -- R and S are on CH
  r_on_ch : ∃ t : ℝ, R = (1 - t) • C + t • H
  s_on_ch : ∃ t : ℝ, S = (1 - t) • C + t • H
  -- M is midpoint of AB
  midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Given side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 21
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 29

/-- The length of RS in the special triangle is 4 --/
theorem rs_length (t : SpecialTriangle) : Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 4 := by
  sorry


end rs_length_l2832_283200


namespace geometric_progression_with_conditions_l2832_283261

/-- A geometric progression of four terms satisfying specific conditions -/
theorem geometric_progression_with_conditions :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    -- The sequence forms a geometric progression
    (∃ (q : ℝ), b₂ = b₁ * q ∧ b₃ = b₁ * q^2 ∧ b₄ = b₁ * q^3) ∧
    -- The third term is 9 greater than the first term
    b₃ - b₁ = 9 ∧
    -- The second term is 18 greater than the fourth term
    b₂ - b₄ = 18 ∧
    -- The sequence is (3, -6, 12, -24)
    b₁ = 3 ∧ b₂ = -6 ∧ b₃ = 12 ∧ b₄ = -24 :=
by
  sorry

end geometric_progression_with_conditions_l2832_283261


namespace cosine_equality_degrees_l2832_283201

theorem cosine_equality_degrees (n : ℤ) : ∃ n : ℤ, 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) ∧ n = 43 := by
  sorry

end cosine_equality_degrees_l2832_283201


namespace sphere_hemisphere_radius_equality_l2832_283214

/-- The radius of a sphere is equal to the radius of each of two hemispheres 
    that have the same total volume as the original sphere. -/
theorem sphere_hemisphere_radius_equality (r : ℝ) (h : r > 0) : 
  (4 / 3 * Real.pi * r^3) = (2 * (2 / 3 * Real.pi * r^3)) := by
  sorry

#check sphere_hemisphere_radius_equality

end sphere_hemisphere_radius_equality_l2832_283214


namespace fraction_problem_l2832_283203

theorem fraction_problem (n : ℝ) (F : ℝ) (h1 : n = 70.58823529411765) (h2 : 0.85 * F * n = 36) : F = 0.6 := by
  sorry

end fraction_problem_l2832_283203


namespace intersection_implies_sum_l2832_283211

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 8 = 0}
def N (p q : ℝ) : Set ℝ := {x | x^2 - q*x + p = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N p q = {1} → p + q = 19 := by
  sorry

end intersection_implies_sum_l2832_283211


namespace train_passing_pole_time_l2832_283291

/-- Proves that a train 100 meters long, traveling at 72 km/hr, takes 5 seconds to pass a pole -/
theorem train_passing_pole_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 100 ∧ train_speed_kmh = 72 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
sorry

end train_passing_pole_time_l2832_283291


namespace zinc_copper_ratio_is_117_143_l2832_283202

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zincCopperRatio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def givenMixture : Mixture :=
  { totalWeight := 78
  , zincWeight := 35.1 }

/-- Theorem stating that the ratio of zinc to copper in the given mixture is 117:143 -/
theorem zinc_copper_ratio_is_117_143 :
  zincCopperRatio givenMixture = { numerator := 117, denominator := 143 } :=
  sorry

end zinc_copper_ratio_is_117_143_l2832_283202


namespace simplify_trig_expression_l2832_283278

theorem simplify_trig_expression : 
  Real.sqrt (1 + Real.sin 10) + Real.sqrt (1 - Real.sin 10) = -2 * Real.sin 5 := by
  sorry

end simplify_trig_expression_l2832_283278


namespace cookie_distribution_l2832_283286

theorem cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 120)
  (h2 : num_adults = 2)
  (h3 : num_children = 4) :
  (total_cookies - (total_cookies / 3)) / num_children = 20 := by
  sorry

end cookie_distribution_l2832_283286


namespace remainder_777_444_mod_11_l2832_283244

theorem remainder_777_444_mod_11 : 777^444 % 11 = 3 := by
  sorry

end remainder_777_444_mod_11_l2832_283244


namespace quadratic_root_condition_l2832_283215

theorem quadratic_root_condition (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - a*x + 1 = 0) → a > 2 := by
  sorry

end quadratic_root_condition_l2832_283215


namespace olaf_toy_cars_l2832_283258

/-- Represents the toy car collection problem -/
def toy_car_problem (initial_collection : ℕ) (grandpa_factor : ℕ) (dad_gift : ℕ) 
  (mum_dad_diff : ℕ) (auntie_gift : ℕ) (final_total : ℕ) : Prop :=
  ∃ (uncle_gift : ℕ),
    initial_collection + (grandpa_factor * uncle_gift) + uncle_gift + 
    dad_gift + (dad_gift + mum_dad_diff) + auntie_gift = final_total ∧
    auntie_gift - uncle_gift = 1

/-- The specific instance of the toy car problem -/
theorem olaf_toy_cars : 
  toy_car_problem 150 2 10 5 6 196 := by
  sorry

end olaf_toy_cars_l2832_283258


namespace difference_of_squares_l2832_283287

theorem difference_of_squares (x y : ℝ) : (x + y) * (x - y) = x^2 - y^2 := by
  sorry

end difference_of_squares_l2832_283287


namespace p_necessary_not_sufficient_l2832_283283

-- Define propositions p and q
variable (p q : Prop)

-- Define the original implication and its contrapositive
def original_implication := p → q
def contrapositive := ¬q → ¬p

-- Define necessary and sufficient conditions
def necessary (p q : Prop) := q → p
def sufficient (p q : Prop) := p → q

theorem p_necessary_not_sufficient (h1 : ¬(original_implication p q)) (h2 : contrapositive p q) :
  necessary p q ∧ ¬(sufficient p q) := by sorry

end p_necessary_not_sufficient_l2832_283283


namespace complex_fraction_equality_l2832_283293

theorem complex_fraction_equality : (2 : ℂ) / (1 - I) = 1 + I := by sorry

end complex_fraction_equality_l2832_283293


namespace cubic_sum_nonnegative_l2832_283206

theorem cubic_sum_nonnegative (c : ℝ) (X Y : ℝ) 
  (hX : X^2 - c*X - c = 0) 
  (hY : Y^2 - c*Y - c = 0) : 
  X^3 + Y^3 + (X*Y)^3 ≥ 0 := by
  sorry

end cubic_sum_nonnegative_l2832_283206


namespace dorchester_washed_16_puppies_l2832_283246

/-- Represents the number of puppies washed by Dorchester on Wednesday -/
def puppies_washed : ℕ := sorry

/-- Dorchester's daily base pay in cents -/
def daily_base_pay : ℕ := 4000

/-- Pay per puppy washed in cents -/
def pay_per_puppy : ℕ := 225

/-- Total earnings on Wednesday in cents -/
def total_earnings : ℕ := 7600

/-- Theorem stating that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_washed_16_puppies : 
  puppies_washed = 16 ∧
  total_earnings = daily_base_pay + puppies_washed * pay_per_puppy :=
sorry

end dorchester_washed_16_puppies_l2832_283246


namespace cassy_jars_left_l2832_283223

/-- The number of jars left unpacked when Cassy fills all boxes -/
def jars_left_unpacked (jars_per_box1 : ℕ) (num_boxes1 : ℕ) 
                       (jars_per_box2 : ℕ) (num_boxes2 : ℕ) 
                       (total_jars : ℕ) : ℕ :=
  total_jars - (jars_per_box1 * num_boxes1 + jars_per_box2 * num_boxes2)

theorem cassy_jars_left :
  jars_left_unpacked 12 10 10 30 500 = 80 := by
  sorry

end cassy_jars_left_l2832_283223


namespace seven_story_pagoda_top_lanterns_l2832_283294

/-- Represents a pagoda with a given number of stories and lanterns -/
structure Pagoda where
  stories : ℕ
  total_lanterns : ℕ
  lanterns_ratio : ℕ -- ratio of lanterns between adjacent stories

/-- Calculates the number of lanterns on the top story of a pagoda -/
def top_story_lanterns (p : Pagoda) : ℕ :=
  sorry

/-- Theorem: For a 7-story pagoda with a lantern ratio of 2 and 381 total lanterns,
    the number of lanterns on the top story is 3 -/
theorem seven_story_pagoda_top_lanterns :
  let p : Pagoda := { stories := 7, total_lanterns := 381, lanterns_ratio := 2 }
  top_story_lanterns p = 3 := by
  sorry

end seven_story_pagoda_top_lanterns_l2832_283294


namespace clock_angle_at_6_30_l2832_283274

/-- The smaller angle between the hour and minute hands of a clock at 6:30 -/
def clock_angle : ℝ :=
  let hour_hand_rate : ℝ := 0.5  -- degrees per minute
  let minute_hand_rate : ℝ := 6  -- degrees per minute
  let time_passed : ℝ := 30      -- minutes since 6:00
  let hour_hand_position : ℝ := hour_hand_rate * time_passed
  let minute_hand_position : ℝ := minute_hand_rate * time_passed
  minute_hand_position - hour_hand_position

theorem clock_angle_at_6_30 : clock_angle = 15 := by
  sorry

end clock_angle_at_6_30_l2832_283274


namespace cube_edge_length_l2832_283219

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (cube_edge : ℝ) : 
  box_edge = 1 →
  num_cubes = 8 →
  num_cubes = (box_edge / cube_edge) ^ 3 →
  cube_edge * 100 = 50 := by
  sorry

end cube_edge_length_l2832_283219


namespace company_workers_l2832_283272

theorem company_workers (total : ℕ) (men : ℕ) (women : ℕ) : 
  (3 * total / 10 = men) →  -- One-third without plan, 40% of those with plan are men
  (3 * total / 5 = men + women) →  -- Total workers
  (men = 120) →  -- Given number of men
  (women = 180) :=
sorry

end company_workers_l2832_283272


namespace symmetry_implies_values_l2832_283232

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The function g(x) = x^2 + ax + b -/
def g (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The condition of symmetry about x = 1 -/
def symmetry_condition (a b : ℝ) : Prop :=
  ∀ x, g a b x = f (2 - x)

/-- Theorem: If f and g are symmetrical about x = 1, then a = -4 and b = 4 -/
theorem symmetry_implies_values :
  ∀ a b, symmetry_condition a b → a = -4 ∧ b = 4 := by
  sorry

end symmetry_implies_values_l2832_283232


namespace one_root_quadratic_l2832_283281

theorem one_root_quadratic (k : ℝ) : 
  (∃! x : ℝ, k * x^2 - 8 * x + 16 = 0) → k = 0 ∨ k = 1 := by
  sorry

end one_root_quadratic_l2832_283281


namespace angle_measure_l2832_283229

theorem angle_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (180 - x) = 3 * x + 10 → x = 42.5 := by
  sorry

end angle_measure_l2832_283229


namespace inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l2832_283234

theorem inequality_upper_bound (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 4 / Real.sqrt 3 :=
by sorry

theorem upper_bound_tight : 
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) = 4 / Real.sqrt 3 :=
by sorry

theorem smallest_upper_bound :
  ∀ M : ℝ, M < 4 / Real.sqrt 3 →
  ∃ (x y z w : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > M :=
by sorry

end inequality_upper_bound_upper_bound_tight_smallest_upper_bound_l2832_283234


namespace vector_magnitude_l2832_283220

def a : ℝ × ℝ := (-2, -1)

theorem vector_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10) 
  (h2 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5) : 
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end vector_magnitude_l2832_283220


namespace root_sum_reciprocal_l2832_283284

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 24*x^2 + 88*x - 75 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 88*s - 75) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by
  sorry

#check root_sum_reciprocal

end root_sum_reciprocal_l2832_283284


namespace baker_cakes_problem_l2832_283292

theorem baker_cakes_problem (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) 
  (pastry_cake_difference : ℕ) :
  pastries_made = 131 →
  cakes_sold = 70 →
  pastries_sold = 88 →
  pastry_cake_difference = 112 →
  ∃ cakes_made : ℕ, 
    cakes_made + pastry_cake_difference = pastries_made ∧
    cakes_made = 107 :=
by
  sorry

end baker_cakes_problem_l2832_283292


namespace at_least_one_not_less_than_six_l2832_283273

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + 4/b < 6 ∧ b + 9/c < 6 ∧ c + 16/a < 6) := by
  sorry

end at_least_one_not_less_than_six_l2832_283273


namespace inverse_proportion_problem_l2832_283239

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 30)
  (h4 : y₁ = 8)
  (h5 : y₂ = 24) :
  x₂ = 10 := by
  sorry

end inverse_proportion_problem_l2832_283239


namespace min_rooms_correct_min_rooms_optimal_l2832_283216

/-- The minimum number of rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  if k % 2 = 0
  then 100 * (k / 2 + 1)
  else 100 * ((k - 1) / 2 + 1) + 1

/-- Theorem stating the minimum number of rooms required for 100 tourists -/
theorem min_rooms_correct (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
    ∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

/-- Theorem stating the optimality of the minimum number of rooms -/
theorem min_rooms_optimal (k : ℕ) :
  ∀ n : ℕ, n < min_rooms k →
    ¬∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

end min_rooms_correct_min_rooms_optimal_l2832_283216


namespace a_14_mod_7_l2832_283285

/-- Sequence defined recursively -/
def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1  -- We assume a₁ = 1 based on the solution
  | 2 => 2
  | (n + 3) => a (n + 1) + (a (n + 2))^2

/-- The 14th term of the sequence is congruent to 5 modulo 7 -/
theorem a_14_mod_7 : a 14 ≡ 5 [ZMOD 7] := by
  sorry

end a_14_mod_7_l2832_283285


namespace solution_problem_l2832_283204

theorem solution_problem : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ x + y + x * y = 80 ∧ x = 26 := by
  sorry

end solution_problem_l2832_283204


namespace first_house_price_correct_l2832_283249

/-- Represents the price of Tommy's first house in dollars -/
def first_house_price : ℝ := 400000

/-- Represents the price of Tommy's new house in dollars -/
def new_house_price : ℝ := 500000

/-- Represents the loan percentage for the new house -/
def loan_percentage : ℝ := 0.75

/-- Represents the annual interest rate for the loan -/
def annual_interest_rate : ℝ := 0.035

/-- Represents the loan term in years -/
def loan_term : ℕ := 15

/-- Represents the annual property tax rate -/
def property_tax_rate : ℝ := 0.015

/-- Represents the annual home insurance cost in dollars -/
def annual_insurance_cost : ℝ := 7500

/-- Theorem stating that the first house price is correct given the conditions -/
theorem first_house_price_correct :
  first_house_price = new_house_price / 1.25 ∧
  new_house_price = first_house_price * 1.25 ∧
  loan_percentage * new_house_price * annual_interest_rate +
  property_tax_rate * new_house_price +
  annual_insurance_cost =
  28125 :=
sorry

end first_house_price_correct_l2832_283249


namespace inequality_implication_l2832_283269

theorem inequality_implication (x y : ℝ) (h : x > y) : (1/2 : ℝ)^x < (1/2 : ℝ)^y := by
  sorry

end inequality_implication_l2832_283269


namespace arrange_85550_eq_16_l2832_283260

/-- The number of ways to arrange the digits of 85550 to form a 5-digit number -/
def arrange_85550 : ℕ :=
  let digits : Multiset ℕ := {8, 5, 5, 5, 0}
  let total_digits : ℕ := 5
  let non_zero_digits : ℕ := 4
  16

/-- Theorem stating that the number of ways to arrange the digits of 85550
    to form a 5-digit number is 16 -/
theorem arrange_85550_eq_16 : arrange_85550 = 16 := by
  sorry

end arrange_85550_eq_16_l2832_283260


namespace factorial_ratio_50_48_l2832_283268

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_50_48_l2832_283268


namespace kids_difference_l2832_283275

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 6) 
  (h2 : tuesday = 5) : 
  monday - tuesday = 1 := by
  sorry

end kids_difference_l2832_283275


namespace complex_division_result_l2832_283280

theorem complex_division_result : 
  let z : ℂ := (3 + 7*I) / I
  (z.re = 7) ∧ (z.im = -3) := by sorry

end complex_division_result_l2832_283280


namespace parabola_properties_l2832_283222

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 2)

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := 2

theorem parabola_properties :
  (∀ x : ℝ, parabola x = -2 * (x - 2)^2 + 2) ∧
  (vertex = (2, 2)) ∧
  (axis_of_symmetry = 2) ∧
  (∀ x : ℝ, x ≥ 2 → ∀ y : ℝ, y > x → parabola y < parabola x) :=
by sorry

end parabola_properties_l2832_283222


namespace kittens_at_shelter_l2832_283243

def number_of_puppies : ℕ := 32

def number_of_kittens : ℕ := 2 * number_of_puppies + 14

theorem kittens_at_shelter : number_of_kittens = 78 := by
  sorry

end kittens_at_shelter_l2832_283243


namespace inequality_proof_l2832_283241

theorem inequality_proof (p q r : ℝ) (n : ℕ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hpqr : p * q * r = 1) :
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end inequality_proof_l2832_283241


namespace real_y_condition_l2832_283289

theorem real_y_condition (x y : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 5 * x * y + x + 7 = 0) ↔ (x ≤ -6/5 ∨ x ≥ 14/5) :=
by sorry

end real_y_condition_l2832_283289


namespace problem1_l2832_283209

theorem problem1 (m n : ℝ) : (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

end problem1_l2832_283209


namespace skaters_practice_hours_l2832_283277

/-- Represents the practice hours for each skater -/
structure SkaterHours where
  hannah_weekend : ℕ
  hannah_weekday : ℕ
  sarah_weekday : ℕ
  sarah_weekend : ℕ
  emma_weekday : ℕ
  emma_weekend : ℕ

/-- Calculates the total practice hours for all skaters -/
def total_practice_hours (hours : SkaterHours) : ℕ :=
  hours.hannah_weekend + hours.hannah_weekday +
  hours.sarah_weekday + hours.sarah_weekend +
  hours.emma_weekday + hours.emma_weekend

/-- Theorem stating the total practice hours for the skaters -/
theorem skaters_practice_hours :
  ∃ (hours : SkaterHours),
    hours.hannah_weekend = 8 ∧
    hours.hannah_weekday = hours.hannah_weekend + 17 ∧
    hours.sarah_weekday = 12 ∧
    hours.sarah_weekend = 6 ∧
    hours.emma_weekday = 2 * hours.sarah_weekday ∧
    hours.emma_weekend = hours.sarah_weekend + 5 ∧
    total_practice_hours hours = 86 := by
  sorry

end skaters_practice_hours_l2832_283277


namespace fifteenth_triangular_number_is_120_and_even_l2832_283225

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 and it is even -/
theorem fifteenth_triangular_number_is_120_and_even :
  triangular_number 15 = 120 ∧ Even (triangular_number 15) := by
  sorry

end fifteenth_triangular_number_is_120_and_even_l2832_283225


namespace decagon_triangle_count_l2832_283290

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) where
  (polygon : RegularPolygon n)
  (v1 v2 v3 : Fin n)
  (distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1)

/-- Two triangles in a regular polygon are congruent if they have the same shape -/
def CongruentTriangles (n : ℕ) (t1 t2 : PolygonTriangle n) : Prop :=
  sorry

/-- The number of non-congruent triangles in a regular decagon -/
def NumNonCongruentTriangles (p : RegularPolygon 10) : ℕ :=
  sorry

theorem decagon_triangle_count :
  ∀ (p : RegularPolygon 10), NumNonCongruentTriangles p = 8 :=
sorry

end decagon_triangle_count_l2832_283290


namespace triangle_problem_l2832_283242

-- Define the triangles and their properties
def Triangle (A B C : ℝ × ℝ) := True

def is_45_45_90_triangle (A B D : ℝ × ℝ) : Prop :=
  Triangle A B D ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2

def is_30_60_90_triangle (A C D : ℝ × ℝ) : Prop :=
  Triangle A C D ∧ 
  4 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * ((D.1 - A.1)^2 + (D.2 - A.2)^2)

-- Define the theorem
theorem triangle_problem (A B C D : ℝ × ℝ) :
  is_45_45_90_triangle A B D →
  is_30_60_90_triangle A C D →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 24 :=
by
  sorry


end triangle_problem_l2832_283242


namespace meals_without_restrictions_l2832_283262

theorem meals_without_restrictions (total clients vegan kosher gluten_free vegan_kosher vegan_gluten_free kosher_gluten_free vegan_kosher_gluten_free : ℕ) 
  (h1 : total = 50)
  (h2 : vegan = 10)
  (h3 : kosher = 12)
  (h4 : gluten_free = 6)
  (h5 : vegan_kosher = 3)
  (h6 : vegan_gluten_free = 4)
  (h7 : kosher_gluten_free = 2)
  (h8 : vegan_kosher_gluten_free = 1) :
  total - (vegan + kosher + gluten_free - vegan_kosher - vegan_gluten_free - kosher_gluten_free + vegan_kosher_gluten_free) = 30 := by
  sorry

end meals_without_restrictions_l2832_283262


namespace solve_exponential_equation_l2832_283228

theorem solve_exponential_equation :
  ∃! x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) :=
by
  use -10
  sorry

end solve_exponential_equation_l2832_283228


namespace carson_gold_stars_l2832_283265

/-- Proves that Carson earned 6 gold stars yesterday -/
theorem carson_gold_stars :
  ∀ (yesterday today total : ℕ),
    today = 9 →
    total = 15 →
    total = yesterday + today →
    yesterday = 6 := by
  sorry

end carson_gold_stars_l2832_283265


namespace age_difference_l2832_283213

/-- Proves that Sachin is 8 years younger than Rahul given their age ratio and Sachin's age -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 28 →
  (sachin_age : ℚ) / rahul_age = 7 / 9 →
  rahul_age - sachin_age = 8 := by
sorry

end age_difference_l2832_283213


namespace candy_exchange_theorem_l2832_283245

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem candy_exchange_theorem :
  (choose 7 5) * (choose 9 5) = 2646 := by sorry

end candy_exchange_theorem_l2832_283245


namespace area_of_special_quadrilateral_in_cube_l2832_283298

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Calculate the area of a quadrilateral given its vertices -/
def areaOfQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is a vertex of a cube -/
def isVertexOfCube (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if a point is a midpoint of an edge of a cube -/
def isMidpointOfCubeEdge (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if two points are diagonally opposite vertices of a cube -/
def areDiagonallyOppositeVertices (p1 p2 : Point3D) (cube : Cube) : Prop := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral_in_cube (cube : Cube) (a b c d : Point3D) :
  cube.sideLength = 2 →
  isVertexOfCube a cube →
  isVertexOfCube c cube →
  isMidpointOfCubeEdge b cube →
  isMidpointOfCubeEdge d cube →
  areDiagonallyOppositeVertices a c cube →
  areaOfQuadrilateral ⟨a, b, c, d⟩ = 2 * Real.sqrt 6 := by
  sorry

end area_of_special_quadrilateral_in_cube_l2832_283298


namespace sam_spent_three_dimes_per_candy_bar_l2832_283236

/-- Represents the number of cents in a dime -/
def dime_value : ℕ := 10

/-- Represents the number of cents in a quarter -/
def quarter_value : ℕ := 25

/-- Represents the initial number of dimes Sam has -/
def initial_dimes : ℕ := 19

/-- Represents the initial number of quarters Sam has -/
def initial_quarters : ℕ := 6

/-- Represents the number of candy bars Sam buys -/
def candy_bars : ℕ := 4

/-- Represents the number of lollipops Sam buys -/
def lollipops : ℕ := 1

/-- Represents the amount of money Sam has left after purchases (in cents) -/
def money_left : ℕ := 195

/-- Proves that Sam spent 3 dimes on each candy bar -/
theorem sam_spent_three_dimes_per_candy_bar :
  ∃ (dimes_per_candy : ℕ),
    dimes_per_candy * candy_bars * dime_value + 
    lollipops * quarter_value + 
    money_left = 
    initial_dimes * dime_value + 
    initial_quarters * quarter_value ∧
    dimes_per_candy = 3 := by
  sorry

end sam_spent_three_dimes_per_candy_bar_l2832_283236


namespace line_intercept_product_l2832_283212

/-- Given a line 8x + 5y + c = 0, if the product of its x-intercept and y-intercept is 24,
    then c = ±8√15 -/
theorem line_intercept_product (c : ℝ) : 
  (∃ x y : ℝ, 8*x + 5*y + c = 0 ∧ x * y = 24) → 
  (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) := by
sorry

end line_intercept_product_l2832_283212


namespace min_value_expression_l2832_283233

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (min : ℝ), min = Real.sqrt 10 + Real.sqrt 5 ∧
  ∀ (x : ℝ), (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ x → x ≤ min :=
by sorry

end min_value_expression_l2832_283233


namespace smallest_value_u3_plus_v3_l2832_283210

theorem smallest_value_u3_plus_v3 (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 11) : 
  Complex.abs (u^3 + v^3) ≥ 14.5 := by
sorry

end smallest_value_u3_plus_v3_l2832_283210


namespace product_evaluation_l2832_283299

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n^2 + 1) = 7200 := by
  sorry

end product_evaluation_l2832_283299


namespace octal_135_equals_binary_1011101_l2832_283288

-- Define a function to convert octal to binary
def octal_to_binary (octal : ℕ) : ℕ := sorry

-- State the theorem
theorem octal_135_equals_binary_1011101 :
  octal_to_binary 135 = 1011101 := by sorry

end octal_135_equals_binary_1011101_l2832_283288


namespace min_reciprocal_sum_l2832_283266

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
sorry

end min_reciprocal_sum_l2832_283266


namespace attendees_equal_22_l2832_283253

/-- Represents the total number of people who attended a performance given ticket prices and total revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - num_children * child_price) / adult_price
  num_adults + num_children

/-- Theorem stating that given the specific conditions, the total number of attendees is 22 --/
theorem attendees_equal_22 :
  total_attendees 8 1 50 18 = 22 := by
  sorry

end attendees_equal_22_l2832_283253


namespace problem_solution_l2832_283256

theorem problem_solution (a b c : ℝ) 
  (sum_eq : a + b + c = 99)
  (equal_after_change : a + 6 = b - 6 ∧ b - 6 = 5 * c) : 
  b = 51 := by
sorry

end problem_solution_l2832_283256


namespace students_in_all_activities_l2832_283295

theorem students_in_all_activities (total : ℕ) (chess : ℕ) (music : ℕ) (art : ℕ) (at_least_two : ℕ) :
  total = 25 →
  chess = 12 →
  music = 15 →
  art = 11 →
  at_least_two = 11 →
  ∃ (only_chess only_music only_art chess_music chess_art music_art all_three : ℕ),
    only_chess + only_music + only_art + chess_music + chess_art + music_art + all_three = total ∧
    only_chess + chess_music + chess_art + all_three = chess ∧
    only_music + chess_music + music_art + all_three = music ∧
    only_art + chess_art + music_art + all_three = art ∧
    chess_music + chess_art + music_art + all_three = at_least_two ∧
    all_three = 4 :=
by sorry

end students_in_all_activities_l2832_283295


namespace largest_package_size_l2832_283257

theorem largest_package_size (a b c : ℕ) (ha : a = 30) (hb : b = 45) (hc : c = 75) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end largest_package_size_l2832_283257


namespace min_value_theorem_l2832_283251

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 2/y + 3/z = 1) :
  x + y/2 + z/3 ≥ 9 ∧ (x + y/2 + z/3 = 9 ↔ x = 3 ∧ y = 6 ∧ z = 9) := by
  sorry

end min_value_theorem_l2832_283251


namespace probability_three_common_books_is_32_495_l2832_283218

def total_books : ℕ := 12
def books_selected : ℕ := 4

def probability_three_common_books : ℚ :=
  (Nat.choose total_books 3 * Nat.choose (total_books - 3) 1 * Nat.choose (total_books - 4) 1) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books_is_32_495 :
  probability_three_common_books = 32 / 495 := by
  sorry

end probability_three_common_books_is_32_495_l2832_283218
