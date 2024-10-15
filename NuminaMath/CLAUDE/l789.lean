import Mathlib

namespace NUMINAMATH_CALUDE_y_influenced_by_other_factors_other_factors_lead_to_random_errors_l789_78937

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ      -- Random error

/-- Definition of the linear regression model equation -/
def model_equation (m : LinearRegressionModel) : ℝ → ℝ :=
  fun x => m.b * x + m.a + m.e

/-- Theorem stating that y is influenced by factors other than x -/
theorem y_influenced_by_other_factors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.y m.x ≠ m.b * m.x + m.a :=
sorry

/-- Theorem stating that other factors can lead to random errors -/
theorem other_factors_lead_to_random_errors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.e ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_y_influenced_by_other_factors_other_factors_lead_to_random_errors_l789_78937


namespace NUMINAMATH_CALUDE_train_length_l789_78950

/-- The length of a train given its speed, the speed of a man walking in the same direction,
    and the time it takes for the train to cross the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 71.99424046076314 →
  (train_speed - man_speed) * (5 / 18) * crossing_time = 1199.9040076793857 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l789_78950


namespace NUMINAMATH_CALUDE_average_value_of_series_l789_78907

theorem average_value_of_series (z : ℝ) :
  let series := [4*z, 6*z, 9*z, 13.5*z, 20.25*z]
  (series.sum / series.length : ℝ) = 10.55 * z :=
by sorry

end NUMINAMATH_CALUDE_average_value_of_series_l789_78907


namespace NUMINAMATH_CALUDE_fraction_simplification_l789_78904

theorem fraction_simplification (a b : ℝ) (h1 : b ≠ 0) (h2 : a ≠ b) :
  (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l789_78904


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l789_78941

theorem imaginary_part_of_complex_number : 
  let z : ℂ := 1 / (2 + Complex.I)^2
  Complex.im z = -4/25 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l789_78941


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l789_78951

-- Define proposition p
def p : Prop := ∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ∧ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a^(1/2) > b^(1/2) ↔ Real.log a > Real.log b

theorem compound_propositions_truth : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ((¬p) ∨ (¬q)) ∧ (p ∧ (¬q)) := by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l789_78951


namespace NUMINAMATH_CALUDE_kennedy_car_drive_l789_78952

theorem kennedy_car_drive (miles_per_gallon : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_softball : ℝ) (to_restaurant : ℝ) (to_home : ℝ) 
  (to_friend : ℝ) : 
  miles_per_gallon = 19 →
  initial_gas = 2 →
  to_school = 15 →
  to_softball = 6 →
  to_restaurant = 2 →
  to_home = 11 →
  miles_per_gallon * initial_gas = to_school + to_softball + to_restaurant + to_friend + to_home →
  to_friend = 4 := by sorry

end NUMINAMATH_CALUDE_kennedy_car_drive_l789_78952


namespace NUMINAMATH_CALUDE_matchsticks_left_proof_l789_78966

/-- The number of matchsticks left in the box after Elvis and Ralph make their squares -/
def matchsticks_left (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  total - (elvis_squares * elvis_per_square + ralph_squares * ralph_per_square)

/-- Theorem stating that 6 matchsticks will be left in the box -/
theorem matchsticks_left_proof :
  matchsticks_left 50 5 3 4 8 = 6 := by
  sorry

#eval matchsticks_left 50 5 3 4 8

end NUMINAMATH_CALUDE_matchsticks_left_proof_l789_78966


namespace NUMINAMATH_CALUDE_ellipse_point_coordinates_l789_78942

theorem ellipse_point_coordinates (x y α : ℝ) : 
  x = 4 * Real.cos α → 
  y = 2 * Real.sqrt 3 * Real.sin α → 
  x > 0 → 
  y > 0 → 
  y / x = Real.sqrt 3 → 
  (x, y) = (4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5) := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_coordinates_l789_78942


namespace NUMINAMATH_CALUDE_tangent_angle_range_l789_78953

open Real

noncomputable def curve (x : ℝ) : ℝ := 4 / (exp x + 1)

theorem tangent_angle_range :
  ∀ (x : ℝ), 
  let y := curve x
  let α := Real.arctan (deriv curve x)
  3 * π / 4 ≤ α ∧ α < π :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l789_78953


namespace NUMINAMATH_CALUDE_rectangle_area_arithmetic_progression_l789_78940

/-- The area of a rectangle with sides in arithmetic progression -/
theorem rectangle_area_arithmetic_progression (a d : ℚ) :
  let shorter_side := a
  let longer_side := a + d
  shorter_side > 0 → longer_side > shorter_side →
  (shorter_side * longer_side : ℚ) = a^2 + a*d :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_arithmetic_progression_l789_78940


namespace NUMINAMATH_CALUDE_eight_people_arrangement_l789_78933

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem eight_people_arrangement :
  acceptableArrangements 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangement_l789_78933


namespace NUMINAMATH_CALUDE_money_distribution_l789_78969

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let adriano_initial := 5 * x
  let bruno_initial := 4 * x
  let cesar_initial := 3 * x
  let total_initial := adriano_initial + bruno_initial + cesar_initial
  let daniel_received := x + x + x
  daniel_received / total_initial = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l789_78969


namespace NUMINAMATH_CALUDE_star_seven_three_l789_78997

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 5*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l789_78997


namespace NUMINAMATH_CALUDE_stone_minimum_speed_l789_78919

/-- The minimum speed for a stone to pass through both corners of a building without touching the roof -/
theorem stone_minimum_speed (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  ∃ v₀ : ℝ, v₀ > 0 ∧
    v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧
    ∀ v : ℝ, v > v₀ →
      ∃ (x y : ℝ → ℝ), (∀ t, x t = v * Real.cos α * t ∧ y t = -g * t^2 / 2 + v * Real.sin α * t) ∧
        (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ = 0 ∧ y t₁ = H ∧ x t₂ = l ∧ y t₂ = H - l * Real.tan α) ∧
        (∀ t, 0 ≤ x t ∧ x t ≤ l → y t ≥ H - (x t) * Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_stone_minimum_speed_l789_78919


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l789_78938

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l789_78938


namespace NUMINAMATH_CALUDE_apples_needed_proof_l789_78979

/-- The number of additional apples Tessa needs to make a pie -/
def additional_apples_needed (initial : ℕ) (received : ℕ) (required : ℕ) : ℕ :=
  required - (initial + received)

/-- Theorem: Given Tessa's initial apples, apples received from Anita, and apples needed for a pie,
    the number of additional apples needed is equal to the apples required for a pie
    minus the sum of initial apples and received apples. -/
theorem apples_needed_proof (initial : ℕ) (received : ℕ) (required : ℕ)
    (h1 : initial = 4)
    (h2 : received = 5)
    (h3 : required = 10) :
  additional_apples_needed initial received required = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_needed_proof_l789_78979


namespace NUMINAMATH_CALUDE_assemble_cook_time_is_five_l789_78915

/-- The time it takes to assemble and cook one omelet -/
def assemble_cook_time (
  pepper_chop_time : ℕ)  -- Time to chop one pepper
  (onion_chop_time : ℕ)  -- Time to chop one onion
  (cheese_grate_time : ℕ)  -- Time to grate cheese for one omelet
  (num_peppers : ℕ)  -- Number of peppers to chop
  (num_onions : ℕ)  -- Number of onions to chop
  (num_omelets : ℕ)  -- Number of omelets to make
  (total_time : ℕ)  -- Total time for preparing and cooking all omelets
  : ℕ :=
  let prep_time := pepper_chop_time * num_peppers + onion_chop_time * num_onions + cheese_grate_time * num_omelets
  (total_time - prep_time) / num_omelets

/-- Theorem stating that it takes 5 minutes to assemble and cook one omelet -/
theorem assemble_cook_time_is_five :
  assemble_cook_time 3 4 1 4 2 5 50 = 5 := by
  sorry


end NUMINAMATH_CALUDE_assemble_cook_time_is_five_l789_78915


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l789_78939

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l789_78939


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l789_78978

/-- The coefficient of x^3 in the expansion of (2x + √x)^5 is 10 -/
theorem coefficient_x_cubed_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l789_78978


namespace NUMINAMATH_CALUDE_tuesday_temperature_l789_78903

/-- Given the average temperatures for different sets of days and the temperature on Friday,
    prove the temperature on Tuesday. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : (t + w + th) / 3 = 52)
  (avg_wed_thurs_fri : (w + th + 53) / 3 = 54)
  (fri_temp : ℝ)
  (h_fri_temp : fri_temp = 53) :
  t = 47 :=
by sorry


end NUMINAMATH_CALUDE_tuesday_temperature_l789_78903


namespace NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l789_78998

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 50  -- The row with 51 numbers corresponds to (x+y)^50
  let k : ℕ := 2   -- The third number (0-indexed) corresponds to k=2
  Nat.choose n k = 1225 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_51st_row_third_number_l789_78998


namespace NUMINAMATH_CALUDE_max_intersection_points_l789_78987

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two polygons in a plane -/
structure PolygonConfiguration where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  no_shared_segments : Bool
  potentially_intersect : Bool

/-- Theorem: Maximum number of intersection points between two convex polygons -/
theorem max_intersection_points (config : PolygonConfiguration) 
  (h1 : config.Q₁.convex = true)
  (h2 : config.Q₂.convex = true)
  (h3 : config.Q₂.sides ≥ config.Q₁.sides + 3)
  (h4 : config.no_shared_segments = true)
  (h5 : config.potentially_intersect = true) :
  (max_intersections : ℕ) → max_intersections = config.Q₁.sides * config.Q₂.sides :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_l789_78987


namespace NUMINAMATH_CALUDE_interval_of_decrease_l789_78934

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 30*x - 33

-- Theorem stating the interval of decrease
theorem interval_of_decrease :
  ∀ x ∈ (Set.Ioo (-1 : ℝ) 11), (f' x < 0) ∧
  ∀ y ∉ (Set.Ioo (-1 : ℝ) 11), (f' y ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_interval_of_decrease_l789_78934


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l789_78963

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) : a + b > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l789_78963


namespace NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l789_78971

/-- Definition of a Frustum -/
structure Frustum where
  -- A frustum has two parallel faces (base and top)
  has_parallel_faces : Bool
  -- The extensions of lateral edges intersect at a point
  lateral_edges_intersect : Bool
  -- The extensions of waists of lateral faces intersect at a point
  waists_intersect : Bool

/-- Definition of a proposition about frustums -/
structure FrustumProposition where
  statement : String
  is_correct : Bool

/-- The three given propositions -/
def propositions : List FrustumProposition := [
  { statement := "Cutting a pyramid with a plane, the part between the base of the pyramid and the section is a frustum",
    is_correct := false },
  { statement := "A polyhedron with two parallel and similar bases, and all other faces being trapezoids, is a frustum",
    is_correct := false },
  { statement := "A hexahedron with two parallel faces and the other four faces being isosceles trapezoids is a frustum",
    is_correct := false }
]

/-- Theorem: The number of correct propositions is zero -/
theorem number_of_correct_propositions_is_zero :
  (propositions.filter (λ p => p.is_correct)).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_correct_propositions_is_zero_l789_78971


namespace NUMINAMATH_CALUDE_function_intersection_and_tangency_l789_78985

/-- Given two functions f and g, prove that under certain conditions, 
    the coefficients a, b, and c have specific values. -/
theorem function_intersection_and_tangency 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 + a * x)
  (h2 : ∀ x, g x = b * x^2 + c)
  (h3 : f 2 = 0)
  (h4 : g 2 = 0)
  (h5 : (deriv f) 2 = (deriv g) 2) : 
  a = -8 ∧ b = 4 ∧ c = -16 := by
  sorry

end NUMINAMATH_CALUDE_function_intersection_and_tangency_l789_78985


namespace NUMINAMATH_CALUDE_additional_amount_needed_l789_78916

/-- The cost of the perfume --/
def perfume_cost : ℚ := 75

/-- The amount Christian saved --/
def christian_saved : ℚ := 5

/-- The amount Sue saved --/
def sue_saved : ℚ := 7

/-- The number of yards Christian mowed --/
def yards_mowed : ℕ := 6

/-- The price Christian charged per yard --/
def price_per_yard : ℚ := 6

/-- The number of dogs Sue walked --/
def dogs_walked : ℕ := 8

/-- The price Sue charged per dog --/
def price_per_dog : ℚ := 3

/-- The theorem stating the additional amount needed --/
theorem additional_amount_needed : 
  perfume_cost - (christian_saved + sue_saved + yards_mowed * price_per_yard + dogs_walked * price_per_dog) = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_needed_l789_78916


namespace NUMINAMATH_CALUDE_involutive_function_theorem_l789_78945

/-- A function f is involutive if f(f(x)) = x for all x in its domain -/
def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_theorem (a b c d : ℝ) 
    (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) :
    let f := fun x => (2 * a * x + b) / (3 * c * x + 2 * d)
    Involutive f → 2 * a + 2 * d = 0 := by
  sorry


end NUMINAMATH_CALUDE_involutive_function_theorem_l789_78945


namespace NUMINAMATH_CALUDE_sequence_divisibility_l789_78958

theorem sequence_divisibility (n : ℕ) : 
  (∃ k, k > 0 ∧ k * (k + 1) ≤ 14520 ∧ 120 ∣ (k * (k + 1))) ↔ 
  (∃ m, m ≥ 1 ∧ m ≤ 8 ∧ 120 ∣ (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l789_78958


namespace NUMINAMATH_CALUDE_sin_cos_relation_l789_78967

theorem sin_cos_relation (α : ℝ) : 
  2 * Real.sin (α - π/3) = (2 - Real.sqrt 3) * Real.cos α → 
  Real.sin (2*α) + 3 * (Real.cos α)^2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l789_78967


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_in_third_quadrant_l789_78910

theorem cos_alpha_for_point_in_third_quadrant (a : ℝ) (α : ℝ) :
  a < 0 →
  ∃ (P : ℝ × ℝ), P = (3*a, 4*a) ∧ 
  (∃ (r : ℝ), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_in_third_quadrant_l789_78910


namespace NUMINAMATH_CALUDE_brick_width_is_four_l789_78900

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: A brick with length 8 cm, height 2 cm, and surface area 112 cm² has a width of 4 cm -/
theorem brick_width_is_four :
  ∀ w : ℝ, surface_area 8 w 2 = 112 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_four_l789_78900


namespace NUMINAMATH_CALUDE_two_pipes_fill_time_l789_78988

def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem two_pipes_fill_time :
  fill_time 3 12 → fill_time 2 18 :=
by
  sorry

end NUMINAMATH_CALUDE_two_pipes_fill_time_l789_78988


namespace NUMINAMATH_CALUDE_desired_average_grade_l789_78999

def first_test_score : ℚ := 95
def second_test_score : ℚ := 80
def third_test_score : ℚ := 95

def average_grade : ℚ := (first_test_score + second_test_score + third_test_score) / 3

theorem desired_average_grade :
  average_grade = 90 := by sorry

end NUMINAMATH_CALUDE_desired_average_grade_l789_78999


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l789_78984

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3 × 2^n + m, prove that the common ratio is 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ n, S n = 3 * 2^n + m) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n-1)) :
  ∀ n, n ≥ 2 → a (n+1) / a n = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l789_78984


namespace NUMINAMATH_CALUDE_total_pastries_count_l789_78962

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

theorem total_pastries_count : total_pastries = 73 := by
  sorry

end NUMINAMATH_CALUDE_total_pastries_count_l789_78962


namespace NUMINAMATH_CALUDE_solution_and_inequality_l789_78920

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem solution_and_inequality (t : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 5 = {x | f t (x + 2) ≤ 2}) 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h2 : a + b + c = t) : 
  t = 1 ∧ a^2 / b + b^2 / c + c^2 / a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_and_inequality_l789_78920


namespace NUMINAMATH_CALUDE_new_men_average_age_greater_than_22_l789_78949

theorem new_men_average_age_greater_than_22 
  (A : ℝ) -- Age of the third man who is not replaced
  (B C : ℝ) -- Ages of the two new men
  (h1 : (A + B + C) / 3 > (A + 21 + 23) / 3) -- Average age increases after replacement
  : (B + C) / 2 > 22 := by
sorry

end NUMINAMATH_CALUDE_new_men_average_age_greater_than_22_l789_78949


namespace NUMINAMATH_CALUDE_problem_statement_l789_78993

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else Real.log x / Real.log 0.2

-- Theorem statement
theorem problem_statement (a : ℝ) (h : f (a + 5) = -1) : f a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l789_78993


namespace NUMINAMATH_CALUDE_opposite_sides_m_range_l789_78960

/-- Given two points on opposite sides of a line, prove the range of m -/
theorem opposite_sides_m_range :
  ∀ (m : ℝ),
  (2 * 1 + 3 + m) * (2 * (-4) + (-2) + m) < 0 →
  m ∈ Set.Ioo (-5 : ℝ) 10 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_m_range_l789_78960


namespace NUMINAMATH_CALUDE_max_value_of_expression_l789_78946

/-- The set of digits to be used -/
def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The expression to be maximized -/
def expression (a b c d e f : ℕ) : ℚ :=
  a / b + c / d + e / f

/-- The theorem stating the maximum value of the expression -/
theorem max_value_of_expression :
  ∃ (a b c d e f : ℕ),
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧ f ∈ Digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    expression a b c d e f = 59 / 6 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ Digits → y ∈ Digits → z ∈ Digits → w ∈ Digits → u ∈ Digits → v ∈ Digits →
      x ≠ y → x ≠ z → x ≠ w → x ≠ u → x ≠ v →
      y ≠ z → y ≠ w → y ≠ u → y ≠ v →
      z ≠ w → z ≠ u → z ≠ v →
      w ≠ u → w ≠ v →
      u ≠ v →
      expression x y z w u v ≤ 59 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l789_78946


namespace NUMINAMATH_CALUDE_simplify_expression_l789_78918

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l789_78918


namespace NUMINAMATH_CALUDE_expression_takes_many_values_l789_78968

theorem expression_takes_many_values :
  ∀ (x : ℝ), x ≠ -2 → x ≠ 3 →
  ∃ (y : ℝ), y ≠ x ∧
    (3 + 6 / (3 - x)) ≠ (3 + 6 / (3 - y)) :=
by sorry

end NUMINAMATH_CALUDE_expression_takes_many_values_l789_78968


namespace NUMINAMATH_CALUDE_initial_number_proof_l789_78944

theorem initial_number_proof : ∃ x : ℝ, (x / 34) * 15 + 270 = 405 ∧ x = 306 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l789_78944


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l789_78956

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*k*x + k^2 - k - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (k = 5 → x₁*x₂^2 + x₁^2*x₂ = 190) ∧
  (x₁ - 3*x₂ = 2 → k = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l789_78956


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l789_78943

/-- A right triangle with the given cone volume properties has a hypotenuse of approximately 21.3 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / 3 * π * y^2 * x = 1250 * π) →
  (1 / 3 * π * x^2 * y = 2700 * π) →
  abs (Real.sqrt (x^2 + y^2) - 21.3) < 0.1 := by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l789_78943


namespace NUMINAMATH_CALUDE_exists_M_with_properties_l789_78917

def is_valid_last_four_digits (n : ℕ) : Prop :=
  n < 10000 ∧ 
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 4*a - 3*b ∧
  (n / 1000 ≠ (n / 100) % 10) ∧
  (n / 1000 ≠ (n / 10) % 10) ∧
  (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ (n / 10) % 10) ∧
  (n / 100 % 10 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem exists_M_with_properties :
  ∃ (M : ℕ), 
    M % 8 = 0 ∧
    M % 16 ≠ 0 ∧
    is_valid_last_four_digits (M % 10000) ∧
    M % 1000 = 624 :=
sorry

end NUMINAMATH_CALUDE_exists_M_with_properties_l789_78917


namespace NUMINAMATH_CALUDE_ratio_12min_to_1hour_is_1_to_5_l789_78913

/-- The ratio of 12 minutes to 1 hour -/
def ratio_12min_to_1hour : ℚ × ℚ :=
  sorry

/-- One hour in minutes -/
def minutes_per_hour : ℕ := 60

theorem ratio_12min_to_1hour_is_1_to_5 :
  ratio_12min_to_1hour = (1, 5) := by
  sorry

end NUMINAMATH_CALUDE_ratio_12min_to_1hour_is_1_to_5_l789_78913


namespace NUMINAMATH_CALUDE_ladder_length_l789_78982

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) : 
  angle = 60 * π / 180 →
  adjacent = 4.6 →
  Real.cos angle = adjacent / hypotenuse →
  hypotenuse = 9.2 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l789_78982


namespace NUMINAMATH_CALUDE_tangent_line_at_e_l789_78927

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e :
  let x₀ : ℝ := Real.exp 1
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.exp 1 * (1 / Real.exp 1) + Real.log (Real.exp 1)
  (λ x y => y = m * (x - x₀) + y₀) = (λ x y => y = 2 * x - Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_e_l789_78927


namespace NUMINAMATH_CALUDE_pat_earned_stickers_l789_78911

/-- The number of stickers Pat had at the start of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by
  sorry

end NUMINAMATH_CALUDE_pat_earned_stickers_l789_78911


namespace NUMINAMATH_CALUDE_calculation_proof_l789_78976

theorem calculation_proof : (2014 * 2014 + 2012) - 2013 * 2013 = 6039 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l789_78976


namespace NUMINAMATH_CALUDE_radhika_final_game_count_l789_78935

def initial_games_ratio : Rat := 2/3
def christmas_games : Nat := 12
def birthday_games : Nat := 8
def family_gathering_games : Nat := 5
def additional_purchased_games : Nat := 6

theorem radhika_final_game_count :
  let total_gifted_games := christmas_games + birthday_games + family_gathering_games
  let initial_games := (initial_games_ratio * total_gifted_games).floor
  let games_after_gifts := initial_games + total_gifted_games
  let final_game_count := games_after_gifts + additional_purchased_games
  final_game_count = 47 := by
  sorry

end NUMINAMATH_CALUDE_radhika_final_game_count_l789_78935


namespace NUMINAMATH_CALUDE_platform_length_l789_78922

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 300 →
  platform_cross_time = 27 →
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 450 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l789_78922


namespace NUMINAMATH_CALUDE_download_time_is_450_minutes_l789_78970

-- Define the problem parameters
def min_speed : ℝ := 20
def max_speed : ℝ := 40
def avg_speed : ℝ := 30
def program_a_size : ℝ := 450
def program_b_size : ℝ := 240
def program_c_size : ℝ := 120
def mb_per_gb : ℝ := 1000
def seconds_per_minute : ℝ := 60

-- State the theorem
theorem download_time_is_450_minutes :
  let total_size := (program_a_size + program_b_size + program_c_size) * mb_per_gb
  let download_time_seconds := total_size / avg_speed
  let download_time_minutes := download_time_seconds / seconds_per_minute
  download_time_minutes = 450 := by
sorry

end NUMINAMATH_CALUDE_download_time_is_450_minutes_l789_78970


namespace NUMINAMATH_CALUDE_tuition_fee_agreement_percentage_l789_78994

theorem tuition_fee_agreement_percentage (total_parents : ℕ) (disagree_parents : ℕ) 
  (h1 : total_parents = 800) (h2 : disagree_parents = 640) : 
  (total_parents - disagree_parents : ℝ) / total_parents * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tuition_fee_agreement_percentage_l789_78994


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l789_78930

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the different relation for planes and lines
variable (different : Plane → Plane → Prop)
variable (different_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different α γ ∧ different β γ)
  (h_diff_lines : different_line m n ∧ different_line m l ∧ different_line n l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l789_78930


namespace NUMINAMATH_CALUDE_floor_sum_example_l789_78909

theorem floor_sum_example : ⌊(17.2 : ℝ)⌋ + ⌊(-17.2 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l789_78909


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l789_78923

theorem quadratic_roots_condition (a : ℝ) : 
  (a ∈ Set.Ici 2 → ∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∃ a : ℝ, a ∉ Set.Ici 2 ∧ ∃ x : ℝ, x^2 - a*x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l789_78923


namespace NUMINAMATH_CALUDE_special_right_triangle_area_l789_78921

/-- Represents a right triangle with an incircle that evenly trisects a median -/
structure SpecialRightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Incircle radius
  r : ℝ
  -- Median length
  m : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = 24
  trisected_median : m = 3 * r
  area_condition : a * b = 288

/-- The main theorem -/
theorem special_right_triangle_area (t : SpecialRightTriangle) : 
  ∃ (m n : ℕ), t.a * t.b / 2 = m * Real.sqrt n ∧ m = 144 ∧ n = 1 ∧ ¬ ∃ (p : ℕ), Prime p ∧ n % (p^2) = 0 :=
sorry

end NUMINAMATH_CALUDE_special_right_triangle_area_l789_78921


namespace NUMINAMATH_CALUDE_min_guests_for_cheaper_second_planner_l789_78925

/-- Represents the pricing model of an event planner -/
structure EventPlanner where
  flatFee : ℕ
  perGuestFee : ℕ

/-- Calculates the total cost for a given number of guests -/
def totalCost (planner : EventPlanner) (guests : ℕ) : ℕ :=
  planner.flatFee + planner.perGuestFee * guests

/-- Defines the two event planners -/
def planner1 : EventPlanner := { flatFee := 120, perGuestFee := 18 }
def planner2 : EventPlanner := { flatFee := 250, perGuestFee := 15 }

/-- Theorem stating the minimum number of guests for the second planner to be less expensive -/
theorem min_guests_for_cheaper_second_planner :
  ∀ n : ℕ, (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) ∧
           (n < 44 → totalCost planner2 n ≥ totalCost planner1 n) := by
  sorry

end NUMINAMATH_CALUDE_min_guests_for_cheaper_second_planner_l789_78925


namespace NUMINAMATH_CALUDE_subset_of_all_implies_zero_l789_78912

theorem subset_of_all_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_all_implies_zero_l789_78912


namespace NUMINAMATH_CALUDE_x_value_proof_l789_78972

theorem x_value_proof :
  let equation := (2021 / 2022 - 2022 / 2021) + x = 0
  ∃ x, equation ∧ x = 2022 / 2021 - 2021 / 2022 :=
by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l789_78972


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l789_78957

/-- Represents a sculpture made of unit cubes -/
structure Sculpture where
  totalCubes : Nat
  layer1 : Nat
  layer2 : Nat
  layer3 : Nat
  layer4 : Nat

/-- Calculate the exposed surface area of the sculpture -/
def exposedSurfaceArea (s : Sculpture) : Nat :=
  5 * s.layer1 + 4 * s.layer2 + s.layer3 + 3 * s.layer4

/-- The main theorem stating the exposed surface area of the specific sculpture -/
theorem sculpture_surface_area :
  ∃ (s : Sculpture),
    s.totalCubes = 20 ∧
    s.layer1 = 1 ∧
    s.layer2 = 4 ∧
    s.layer3 = 9 ∧
    s.layer4 = 6 ∧
    exposedSurfaceArea s = 48 := by
  sorry


end NUMINAMATH_CALUDE_sculpture_surface_area_l789_78957


namespace NUMINAMATH_CALUDE_remainder_theorem_l789_78914

theorem remainder_theorem : (8 * 20^34 + 3^34) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l789_78914


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l789_78989

theorem rectangular_box_volume (l w h : ℝ) (area1 area2 area3 : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  area1 = l * w →
  area2 = w * h →
  area3 = l * h →
  area1 = 30 →
  area2 = 18 →
  area3 = 10 →
  l * w * h = 90 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l789_78989


namespace NUMINAMATH_CALUDE_pet_parasites_l789_78975

theorem pet_parasites (dog_burrs : ℕ) : ℕ :=
  let dog_ticks := 6 * dog_burrs
  let dog_fleas := 3 * dog_ticks
  let cat_burrs := 2 * dog_burrs
  let cat_ticks := dog_ticks / 3
  let cat_fleas := 4 * cat_ticks
  let total_parasites := dog_burrs + dog_ticks + dog_fleas + cat_burrs + cat_ticks + cat_fleas
  
  by
  -- Assuming dog_burrs = 12
  have h : dog_burrs = 12 := by sorry
  -- Proof goes here
  sorry

-- The theorem states that given the number of burrs on the dog (which we know is 12),
-- we can calculate the total number of parasites on both pets.
-- The proof would show that this total is indeed 444.

end NUMINAMATH_CALUDE_pet_parasites_l789_78975


namespace NUMINAMATH_CALUDE_circle_center_correct_l789_78983

/-- The equation of a circle in the form ax^2 + bx + cy^2 + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct (eq : CircleEquation) :
  eq.a = 1 ∧ eq.b = -10 ∧ eq.c = 1 ∧ eq.d = -4 ∧ eq.e = -20 →
  let center := findCircleCenter eq
  center.x = 5 ∧ center.y = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l789_78983


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l789_78924

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 10*p^3 + 20*p^2 + 15*p + 6 = 0 →
  q^4 + 10*q^3 + 20*q^2 + 15*q + 6 = 0 →
  r^4 + 10*r^3 + 20*r^2 + 15*r + 6 = 0 →
  s^4 + 10*s^3 + 20*s^2 + 15*s + 6 = 0 →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 10/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l789_78924


namespace NUMINAMATH_CALUDE_unique_intersection_point_l789_78995

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1) * x

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x - g m x

theorem unique_intersection_point (m : ℝ) (hm : m ≥ 1) :
  ∃! x, x > 0 ∧ h m x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l789_78995


namespace NUMINAMATH_CALUDE_gnomes_and_ponies_count_l789_78973

/-- Represents the number of gnomes -/
def num_gnomes : ℕ := 12

/-- Represents the number of ponies -/
def num_ponies : ℕ := 3

/-- The total number of heads in the caravan -/
def total_heads : ℕ := 15

/-- The total number of legs in the caravan -/
def total_legs : ℕ := 36

/-- Each gnome has this many legs -/
def gnome_legs : ℕ := 2

/-- Each pony has this many legs -/
def pony_legs : ℕ := 4

theorem gnomes_and_ponies_count :
  (num_gnomes + num_ponies = total_heads) ∧
  (num_gnomes * gnome_legs + num_ponies * pony_legs = total_legs) :=
by sorry

end NUMINAMATH_CALUDE_gnomes_and_ponies_count_l789_78973


namespace NUMINAMATH_CALUDE_abs_neg_nine_l789_78954

theorem abs_neg_nine : |(-9 : ℤ)| = 9 := by sorry

end NUMINAMATH_CALUDE_abs_neg_nine_l789_78954


namespace NUMINAMATH_CALUDE_circle_area_increase_l789_78947

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := r * 2.5
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 8 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l789_78947


namespace NUMINAMATH_CALUDE_identical_solutions_l789_78977

theorem identical_solutions (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 3*x + k) ↔ k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_identical_solutions_l789_78977


namespace NUMINAMATH_CALUDE_circle_area_probability_l789_78902

theorem circle_area_probability (AB : ℝ) (h_AB : AB = 10) : 
  let prob := (Real.sqrt 64 - Real.sqrt 36) / AB
  prob = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_circle_area_probability_l789_78902


namespace NUMINAMATH_CALUDE_square_difference_divisible_by_13_l789_78932

theorem square_difference_divisible_by_13 (a b : ℕ) :
  a ∈ Finset.range 1001 →
  b ∈ Finset.range 1001 →
  a + b = 1001 →
  13 ∣ (a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_divisible_by_13_l789_78932


namespace NUMINAMATH_CALUDE_correct_calculation_l789_78926

theorem correct_calculation : (1/3) + (-1/2) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l789_78926


namespace NUMINAMATH_CALUDE_g_at_negative_two_l789_78908

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

-- State the theorem
theorem g_at_negative_two : g (-2) = 7/3 := by sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l789_78908


namespace NUMINAMATH_CALUDE_log_sum_property_l789_78959

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_log_sum_property_l789_78959


namespace NUMINAMATH_CALUDE_medicine_percentage_l789_78901

/-- Proves that the percentage of income spent on medicines is 15% --/
theorem medicine_percentage (income : ℕ) (household_percent : ℚ) (clothes_percent : ℚ) (savings : ℕ)
  (h1 : income = 90000)
  (h2 : household_percent = 50 / 100)
  (h3 : clothes_percent = 25 / 100)
  (h4 : savings = 9000) :
  (income - (household_percent * income + clothes_percent * income + savings)) / income = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_medicine_percentage_l789_78901


namespace NUMINAMATH_CALUDE_largest_c_for_inequality_l789_78955

theorem largest_c_for_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c = |Real.log (a / b)| ∧
  (∀ x α : ℝ, 0 < |x| → |x| ≤ c → 0 < α → α < 1 →
    a^α * b^(1-α) ≤ a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) ∧
  (∀ c' : ℝ, c' > c →
    ∃ x α : ℝ, 0 < |x| ∧ |x| ≤ c' ∧ 0 < α ∧ α < 1 ∧
      a^α * b^(1-α) > a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_inequality_l789_78955


namespace NUMINAMATH_CALUDE_original_average_rent_l789_78990

theorem original_average_rent
  (num_friends : ℕ)
  (original_rent : ℝ)
  (increased_rent : ℝ)
  (new_average : ℝ)
  (h1 : num_friends = 4)
  (h2 : original_rent = 1250)
  (h3 : increased_rent = 1250 * 1.16)
  (h4 : new_average = 850)
  : (num_friends * new_average - increased_rent + original_rent) / num_friends = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_average_rent_l789_78990


namespace NUMINAMATH_CALUDE_max_square_plots_is_48_l789_78992

/-- Represents the dimensions and constraints of the field --/
structure FieldParameters where
  length : ℝ
  width : ℝ
  pathwayWidth : ℝ
  availableFencing : ℝ

/-- Calculates the maximum number of square test plots --/
def maxSquarePlots (params : FieldParameters) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square test plots --/
theorem max_square_plots_is_48 (params : FieldParameters) :
  params.length = 45 ∧ 
  params.width = 30 ∧ 
  params.pathwayWidth = 5 ∧ 
  params.availableFencing = 2700 →
  maxSquarePlots params = 48 :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_is_48_l789_78992


namespace NUMINAMATH_CALUDE_max_distance_complex_l789_78981

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 9 * Real.sqrt 61 + 81 ∧
    ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((6 + 5*Complex.I)*z^2 - z^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l789_78981


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l789_78996

theorem smallest_number_with_given_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 11 ∧ 
  n % 17 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 13 = 11 ∧ m % 17 = 9 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l789_78996


namespace NUMINAMATH_CALUDE_no_primes_in_range_l789_78906

theorem no_primes_in_range (n : ℕ) (hn : n > 1) : 
  ∀ k ∈ Set.Ioo (n.factorial) (n.factorial + n + 1), ¬ Nat.Prime k := by
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l789_78906


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l789_78936

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  D = E →  -- Isosceles triangle condition
  F = 2 * D - 40 →  -- Relationship between F and D
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = 70 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l789_78936


namespace NUMINAMATH_CALUDE_meaningful_expression_l789_78991

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2023)) ↔ x ≠ 2023 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l789_78991


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l789_78961

/-- Given a sinusoidal function y = a * sin(b * x + c) + d that oscillates between 5 and -3,
    prove that the amplitude a is equal to 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 :=
by sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l789_78961


namespace NUMINAMATH_CALUDE_reciprocal_problem_l789_78928

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 10) : 50 * (1 / x) = 40 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l789_78928


namespace NUMINAMATH_CALUDE_albert_laps_run_l789_78965

/-- The number of times Albert has already run around the track -/
def laps_run : ℕ := 6

/-- The length of the track in meters -/
def track_length : ℕ := 9

/-- The total distance Albert needs to run in meters -/
def total_distance : ℕ := 99

/-- The number of additional laps Albert will run -/
def additional_laps : ℕ := 5

theorem albert_laps_run :
  laps_run * track_length + additional_laps * track_length = total_distance :=
sorry

end NUMINAMATH_CALUDE_albert_laps_run_l789_78965


namespace NUMINAMATH_CALUDE_train_length_l789_78929

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 54 → time = 7 → speed * time * (1000 / 3600) = 105 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l789_78929


namespace NUMINAMATH_CALUDE_equation_transformation_l789_78964

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = x^2 * (y^2 + y - 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l789_78964


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l789_78980

/-- Given 3 bugs, each eating 2 flowers, prove that the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers : 
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l789_78980


namespace NUMINAMATH_CALUDE_two_carp_heavier_than_three_bream_l789_78905

/-- Represents the weight of a fish species -/
structure FishWeight where
  weight : ℝ
  weight_pos : weight > 0

/-- Given that 6 crucian carps are lighter than 5 perches and 6 crucian carps are heavier than 10 breams,
    prove that 2 crucian carp are heavier than 3 breams. -/
theorem two_carp_heavier_than_three_bream 
  (carp perch bream : FishWeight)
  (h1 : 6 * carp.weight < 5 * perch.weight)
  (h2 : 6 * carp.weight > 10 * bream.weight) :
  2 * carp.weight > 3 * bream.weight := by
sorry

end NUMINAMATH_CALUDE_two_carp_heavier_than_three_bream_l789_78905


namespace NUMINAMATH_CALUDE_aloh3_molecular_weight_l789_78974

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (alWeight oWeight hWeight : ℝ) (moles : ℝ) : ℝ :=
  moles * (alWeight + 3 * oWeight + 3 * hWeight)

/-- Theorem stating the molecular weight of 7 moles of Al(OH)3 -/
theorem aloh3_molecular_weight :
  molecularWeight 26.98 16.00 1.01 7 = 546.07 := by
  sorry

end NUMINAMATH_CALUDE_aloh3_molecular_weight_l789_78974


namespace NUMINAMATH_CALUDE_area_of_circles_with_inscribed_rhombus_l789_78948

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of each circle -/
  radius : ℝ
  /-- The diagonal1 is positive -/
  diagonal1_pos : 0 < diagonal1
  /-- The diagonal2 is positive -/
  diagonal2_pos : 0 < diagonal2
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The rhombus is inscribed in the intersection of the circles -/
  inscribed : (diagonal1 / 2) ^ 2 + (diagonal2 / 2) ^ 2 = radius ^ 2

/-- The theorem stating the relationship between the diagonals and the area of the circles -/
theorem area_of_circles_with_inscribed_rhombus 
  (r : InscribedRhombus) 
  (h1 : r.diagonal1 = 6) 
  (h2 : r.diagonal2 = 12) : 
  π * r.radius ^ 2 = (225 / 4) * π := by
sorry

end NUMINAMATH_CALUDE_area_of_circles_with_inscribed_rhombus_l789_78948


namespace NUMINAMATH_CALUDE_inequality_proof_l789_78986

/-- The function f(x) = |x - a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem to be proved -/
theorem inequality_proof (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (h3 : Set.Icc 0 2 = {x : ℝ | f 1 x ≤ 1})
  (h4 : 1/m + 1/(2*n) = 1) :
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l789_78986


namespace NUMINAMATH_CALUDE_new_average_is_250_l789_78931

/-- A salesperson's commission information -/
structure SalesCommission where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculate the new average commission after a big sale -/
def newAverageCommission (sc : SalesCommission) : ℝ :=
  sorry

/-- Theorem stating the new average commission is $250 under given conditions -/
theorem new_average_is_250 (sc : SalesCommission) 
  (h1 : sc.totalSales = 6)
  (h2 : sc.lastCommission = 1000)
  (h3 : sc.averageIncrease = 150) :
  newAverageCommission sc = 250 :=
sorry

end NUMINAMATH_CALUDE_new_average_is_250_l789_78931
