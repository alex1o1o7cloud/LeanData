import Mathlib

namespace snowman_volume_l3149_314961

theorem snowman_volume (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4 / 3) * π * r^3
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8 = (3168 / 3) * π :=
by sorry

end snowman_volume_l3149_314961


namespace commission_calculation_l3149_314945

def base_salary : ℚ := 370
def past_incomes : List ℚ := [406, 413, 420, 436, 395]
def desired_average : ℚ := 500
def total_weeks : ℕ := 7
def past_weeks : ℕ := 5

theorem commission_calculation (base_salary : ℚ) (past_incomes : List ℚ) 
  (desired_average : ℚ) (total_weeks : ℕ) (past_weeks : ℕ) :
  (desired_average * total_weeks - past_incomes.sum - base_salary * total_weeks) / (total_weeks - past_weeks) = 345 :=
by sorry

end commission_calculation_l3149_314945


namespace perp_planes_necessary_not_sufficient_l3149_314979

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem perp_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_subset : subset_line_plane m α) :
  (∀ m α β, perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m α β, perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end perp_planes_necessary_not_sufficient_l3149_314979


namespace power_multiplication_simplification_l3149_314902

theorem power_multiplication_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end power_multiplication_simplification_l3149_314902


namespace yellow_balls_count_l3149_314930

theorem yellow_balls_count (total_balls : ℕ) (yellow_probability : ℚ) 
  (h1 : total_balls = 40)
  (h2 : yellow_probability = 3/10) :
  (yellow_probability * total_balls : ℚ) = 12 := by
  sorry

end yellow_balls_count_l3149_314930


namespace square_perimeter_from_area_l3149_314942

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 400 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 80 := by
sorry

end square_perimeter_from_area_l3149_314942


namespace simplify_and_find_ratio_l3149_314965

theorem simplify_and_find_ratio (k : ℝ) : 
  (6 * k + 18) / 6 = k + 3 ∧ (1 : ℝ) / 3 = 1 / 3 := by
  sorry

end simplify_and_find_ratio_l3149_314965


namespace min_value_of_expression_l3149_314952

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) :
  x + 1/y ≥ 3 ∧ ∃ (x0 y0 : ℝ), x0 > 1 ∧ x0 - y0 = 1 ∧ x0 + 1/y0 = 3 := by
  sorry

end min_value_of_expression_l3149_314952


namespace hyperbola_midpoint_existence_l3149_314973

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end hyperbola_midpoint_existence_l3149_314973


namespace pencils_left_problem_l3149_314915

def pencils_left (total_pencils : ℕ) (num_students : ℕ) : ℕ :=
  total_pencils - (num_students * (total_pencils / num_students))

theorem pencils_left_problem :
  pencils_left 42 12 = 6 :=
by
  sorry

end pencils_left_problem_l3149_314915


namespace max_three_digit_operation_l3149_314931

theorem max_three_digit_operation :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (200 + n) ≤ 2398 :=
by sorry

end max_three_digit_operation_l3149_314931


namespace dave_remaining_tickets_l3149_314955

/-- Given that Dave had 13 tickets initially and used 6 tickets,
    prove that he has 7 tickets left. -/
theorem dave_remaining_tickets :
  let initial_tickets : ℕ := 13
  let used_tickets : ℕ := 6
  initial_tickets - used_tickets = 7 := by
sorry

end dave_remaining_tickets_l3149_314955


namespace a_c_inequality_l3149_314991

theorem a_c_inequality (a c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : c > 1) :
  a * c + 1 < a + c := by
  sorry

end a_c_inequality_l3149_314991


namespace smaller_fraction_l3149_314943

theorem smaller_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end smaller_fraction_l3149_314943


namespace area_difference_is_one_l3149_314988

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define an equilateral triangle with side length 1
def unit_equilateral_triangle : Set (ℝ × ℝ) := sorry

-- Define the region R (union of square and 12 triangles)
def R : Set (ℝ × ℝ) := sorry

-- Define the smallest convex polygon S containing R
def S : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_difference_is_one :
  area (S \ R) = 1 := by sorry

end area_difference_is_one_l3149_314988


namespace power_fraction_equality_l3149_314948

theorem power_fraction_equality : (40 ^ 56) / (10 ^ 28) = 160 ^ 28 := by
  sorry

end power_fraction_equality_l3149_314948


namespace special_triangle_properties_l3149_314947

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties
  angle_sum : A + B + C = π
  side_angle_relation : (3 * b - c) * Real.cos A - a * Real.cos C = 0
  side_a_value : a = 2 * Real.sqrt 3
  area : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2
  angle_product : Real.sin B * Real.sin C = 2 / 3

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  Real.cos t.A = 1 / 3 ∧
  t.b = 3 ∧ t.c = 3 ∧
  Real.tan t.A + Real.tan t.B + Real.tan t.C = 4 * Real.sqrt 2 := by
  sorry

end special_triangle_properties_l3149_314947


namespace two_numbers_with_sum_and_gcd_l3149_314909

theorem two_numbers_with_sum_and_gcd : ∃ (a b : ℕ), a + b = 168 ∧ Nat.gcd a b = 24 := by
  sorry

end two_numbers_with_sum_and_gcd_l3149_314909


namespace carols_age_l3149_314906

theorem carols_age (bob_age carol_age : ℕ) : 
  bob_age + carol_age = 66 →
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 := by
sorry

end carols_age_l3149_314906


namespace aaron_cards_total_l3149_314980

/-- Given that Aaron initially has 5 cards and finds 62 more, 
    prove that he ends up with 67 cards in total. -/
theorem aaron_cards_total (initial_cards : ℕ) (found_cards : ℕ) : 
  initial_cards = 5 → found_cards = 62 → initial_cards + found_cards = 67 := by
  sorry

end aaron_cards_total_l3149_314980


namespace common_chord_theorem_l3149_314941

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def common_chord_line (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem common_chord_theorem :
  ∃ (x y : ℝ), 
    (C1 x y ∧ C2 x y) → 
    (common_chord_line x y ∧ 
     ∃ (x1 y1 x2 y2 : ℝ), 
       C1 x1 y1 ∧ C2 x1 y1 ∧ 
       C1 x2 y2 ∧ C2 x2 y2 ∧ 
       common_chord_line x1 y1 ∧ 
       common_chord_line x2 y2 ∧ 
       ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 24/5) := by
  sorry

end common_chord_theorem_l3149_314941


namespace salary_january_l3149_314974

def employee_salary (jan feb mar apr may jun : ℕ) : Prop :=
  -- Average salary for Jan, Feb, Mar, Apr, May is Rs. 9,000
  (jan + feb + mar + apr + may) / 5 = 9000 ∧
  -- Average salary for Feb, Mar, Apr, May, Jun is Rs. 10,000
  (feb + mar + apr + may + jun) / 5 = 10000 ∧
  -- Employee receives a bonus of Rs. 1,500 in March
  ∃ (base_mar : ℕ), mar = base_mar + 1500 ∧
  -- Employee receives a deduction of Rs. 1,000 in June
  ∃ (base_jun : ℕ), jun = base_jun - 1000 ∧
  -- Salary for May is Rs. 7,500
  may = 7500 ∧
  -- No pay increase or decrease in the given time frame
  ∃ (base : ℕ), feb = base ∧ apr = base ∧ base_mar = base ∧ base_jun = base

theorem salary_january :
  ∀ (jan feb mar apr may jun : ℕ),
  employee_salary jan feb mar apr may jun →
  jan = 4500 :=
by sorry

end salary_january_l3149_314974


namespace remaining_money_l3149_314953

def initial_amount : ℚ := 10
def candy_bar_cost : ℚ := 2
def chocolate_cost : ℚ := 3
def soda_cost : ℚ := 1.5
def gum_cost : ℚ := 1.25

theorem remaining_money :
  initial_amount - candy_bar_cost - chocolate_cost - soda_cost - gum_cost = 2.25 := by
  sorry

end remaining_money_l3149_314953


namespace line_segment_endpoint_l3149_314954

theorem line_segment_endpoint (x : ℝ) :
  x < 0 ∧
  ((x - 1)^2 + (8 - 3)^2).sqrt = 15 →
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end line_segment_endpoint_l3149_314954


namespace inequality_condition_l3149_314987

def f (x : ℝ) := x^2 - 4*x + 3

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x + 3| < a) ↔ b^2 + 2*b + 3 ≤ a :=
by sorry

end inequality_condition_l3149_314987


namespace length_AX_in_tangent_circles_configuration_l3149_314912

/-- Two circles with radii r₁ and r₂ that are externally tangent -/
structure ExternallyTangentCircles (r₁ r₂ : ℝ) :=
  (center_distance : ℝ)
  (tangent_point : ℝ × ℝ)
  (external_tangent_length : ℝ)
  (h_center_distance : center_distance = r₁ + r₂)

/-- The configuration of two externally tangent circles with their common tangents -/
structure TangentCirclesConfiguration (r₁ r₂ : ℝ) extends ExternallyTangentCircles r₁ r₂ :=
  (common_external_tangent_point_A : ℝ × ℝ)
  (common_external_tangent_point_B : ℝ × ℝ)
  (common_internal_tangent_intersection : ℝ × ℝ)

/-- The theorem stating the length of AX in the given configuration -/
theorem length_AX_in_tangent_circles_configuration 
  (config : TangentCirclesConfiguration 20 13) : 
  ∃ (AX : ℝ), AX = 2 * Real.sqrt 65 :=
sorry

end length_AX_in_tangent_circles_configuration_l3149_314912


namespace fixed_point_of_exponential_function_l3149_314968

/-- The function f(x) = 3 + a^(x-1) always passes through the point (1, 4) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f := λ x : ℝ => 3 + a^(x - 1)
  f 1 = 4 := by sorry

end fixed_point_of_exponential_function_l3149_314968


namespace min_value_expression_l3149_314997

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) - x * y * z ≥ 1 / 2 := by
  sorry

end min_value_expression_l3149_314997


namespace two_digit_average_decimal_l3149_314910

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →             -- their average equals the decimal representation
  max m n = 50 :=                         -- the larger of the two is 50
by sorry

end two_digit_average_decimal_l3149_314910


namespace ages_sum_l3149_314914

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 128 → a + b + c = 18 := by
  sorry

end ages_sum_l3149_314914


namespace total_cupcakes_l3149_314970

theorem total_cupcakes (cupcakes_per_event : ℕ) (number_of_events : ℕ) 
  (h1 : cupcakes_per_event = 156) 
  (h2 : number_of_events = 12) : 
  cupcakes_per_event * number_of_events = 1872 := by
sorry

end total_cupcakes_l3149_314970


namespace pencil_discount_l3149_314901

theorem pencil_discount (original_cost final_price : ℝ) 
  (h1 : original_cost = 4)
  (h2 : final_price = 3.37) : 
  original_cost - final_price = 0.63 := by
  sorry

end pencil_discount_l3149_314901


namespace exam_mode_l3149_314919

/-- Represents a score in the music theory exam -/
structure Score where
  value : ℕ
  deriving Repr

/-- Represents the frequency of each score -/
def ScoreFrequency := Score → ℕ

/-- The set of all scores in the exam -/
def ExamScores : Set Score := sorry

/-- The frequency distribution of scores in the exam -/
def examFrequency : ScoreFrequency := sorry

/-- Definition of mode: the score that appears most frequently -/
def isMode (s : Score) (freq : ScoreFrequency) (scores : Set Score) : Prop :=
  ∀ t ∈ scores, freq s ≥ freq t

/-- The mode of the exam scores is 88 -/
theorem exam_mode :
  ∃ s : Score, s.value = 88 ∧ isMode s examFrequency ExamScores := by sorry

end exam_mode_l3149_314919


namespace distinctFourDigitNumbers_eq_360_l3149_314934

/-- The number of distinct four-digit numbers that can be formed using the digits 1, 2, 3, 4, 5,
    where exactly one digit repeats once. -/
def distinctFourDigitNumbers : ℕ :=
  let digits : Fin 5 := 5
  let positionsForRepeatedDigit : ℕ := Nat.choose 4 2
  let remainingDigitChoices : ℕ := 4 * 3
  digits * positionsForRepeatedDigit * remainingDigitChoices

/-- Theorem stating that the number of distinct four-digit numbers under the given conditions is 360. -/
theorem distinctFourDigitNumbers_eq_360 : distinctFourDigitNumbers = 360 := by
  sorry

end distinctFourDigitNumbers_eq_360_l3149_314934


namespace constant_expression_l3149_314975

theorem constant_expression (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 10)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 20) :
  5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4) = 120 := by
  sorry

end constant_expression_l3149_314975


namespace expensive_candy_price_l3149_314932

/-- Given a mixture of two types of candy, prove the price of the more expensive candy. -/
theorem expensive_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheap_price : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheap_price = 2)
  (h4 : cheap_weight = 64) :
  ∃ (expensive_price : ℝ), expensive_price = 3 := by
sorry

end expensive_candy_price_l3149_314932


namespace seashells_after_giving_away_starfish_count_indeterminate_l3149_314938

/-- Proves that the number of seashells after giving some away is correct -/
theorem seashells_after_giving_away 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  final_seashells = initial_seashells - seashells_given_away :=
by sorry

/-- The number of starfish cannot be determined from the given information -/
theorem starfish_count_indeterminate 
  (initial_seashells : ℕ) 
  (seashells_given_away : ℕ) 
  (final_seashells : ℕ) 
  (starfish : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : seashells_given_away = 13)
  (h3 : final_seashells = 36) :
  ∃ (n : ℕ), starfish = n :=
by sorry

end seashells_after_giving_away_starfish_count_indeterminate_l3149_314938


namespace johns_total_time_l3149_314926

/-- Represents the time John spent on various activities related to his travels and book writing --/
structure TravelTime where
  southAmerica : ℕ  -- Time spent exploring South America (in years)
  africa : ℕ        -- Time spent exploring Africa (in years)
  manuscriptTime : ℕ -- Time spent compiling notes into a manuscript (in months)
  editingTime : ℕ   -- Time spent finalizing the book with an editor (in months)

/-- Calculates the total time John spent on his adventures, note writing, and book creation --/
def totalTime (t : TravelTime) : ℕ :=
  -- Convert exploration time to months and add note-writing time
  (t.southAmerica * 12 + t.southAmerica * 6) +
  -- Convert Africa exploration time to months and add note-writing time
  (t.africa * 12 + t.africa * 4) +
  -- Add manuscript compilation and editing time
  t.manuscriptTime + t.editingTime

/-- Theorem stating that John's total time spent is 100 months --/
theorem johns_total_time :
  ∀ t : TravelTime,
    t.southAmerica = 3 ∧
    t.africa = 2 ∧
    t.manuscriptTime = 8 ∧
    t.editingTime = 6 →
    totalTime t = 100 :=
by
  sorry


end johns_total_time_l3149_314926


namespace hyperbola_focal_length_l3149_314936

/-- The focal length of a hyperbola with equation x²/4 - y²/5 = 1 is 6 -/
theorem hyperbola_focal_length : ∃ (a b c : ℝ),
  (a^2 = 4 ∧ b^2 = 5) →
  (c^2 = a^2 + b^2) →
  (2 * c = 6) :=
sorry

end hyperbola_focal_length_l3149_314936


namespace exponent_of_five_in_30_factorial_l3149_314921

theorem exponent_of_five_in_30_factorial : 
  ∃ k : ℕ, (30 : ℕ).factorial = 5^7 * k ∧ k % 5 ≠ 0 := by
  sorry

end exponent_of_five_in_30_factorial_l3149_314921


namespace ratio_IJ_IF_is_14_13_l3149_314923

/-- A structure representing the geometric configuration described in the problem -/
structure TriangleConfiguration where
  /-- Point F -/
  F : ℝ × ℝ
  /-- Point G -/
  G : ℝ × ℝ
  /-- Point H -/
  H : ℝ × ℝ
  /-- Point I -/
  I : ℝ × ℝ
  /-- Point J -/
  J : ℝ × ℝ
  /-- FGH is a right triangle with right angle at H -/
  FGH_right_at_H : (F.1 - H.1) * (G.1 - H.1) + (F.2 - H.2) * (G.2 - H.2) = 0
  /-- FG = 5 -/
  FG_length : (F.1 - G.1)^2 + (F.2 - G.2)^2 = 25
  /-- GH = 12 -/
  GH_length : (G.1 - H.1)^2 + (G.2 - H.2)^2 = 144
  /-- FHI is a right triangle with right angle at F -/
  FHI_right_at_F : (H.1 - F.1) * (I.1 - F.1) + (H.2 - F.2) * (I.2 - F.2) = 0
  /-- FI = 15 -/
  FI_length : (F.1 - I.1)^2 + (F.2 - I.2)^2 = 225
  /-- H and I are on opposite sides of FG -/
  H_I_opposite_sides : ((G.1 - F.1) * (H.2 - F.2) - (G.2 - F.2) * (H.1 - F.1)) *
                       ((G.1 - F.1) * (I.2 - F.2) - (G.2 - F.2) * (I.1 - F.1)) < 0
  /-- IJ is parallel to FG -/
  IJ_parallel_FG : (J.1 - I.1) * (G.2 - F.2) = (J.2 - I.2) * (G.1 - F.1)
  /-- J is on the extension of GH -/
  J_on_GH_extended : ∃ t : ℝ, J.1 = G.1 + t * (H.1 - G.1) ∧ J.2 = G.2 + t * (H.2 - G.2)

/-- The main theorem stating that the ratio IJ/IF is equal to 14/13 -/
theorem ratio_IJ_IF_is_14_13 (config : TriangleConfiguration) :
  let IJ := ((config.I.1 - config.J.1)^2 + (config.I.2 - config.J.2)^2).sqrt
  let IF := ((config.I.1 - config.F.1)^2 + (config.I.2 - config.F.2)^2).sqrt
  IJ / IF = 14 / 13 :=
sorry

end ratio_IJ_IF_is_14_13_l3149_314923


namespace three_legged_dogs_carly_three_legged_dogs_l3149_314992

theorem three_legged_dogs (total_nails : ℕ) (total_dogs : ℕ) : ℕ :=
  let nails_per_paw := 4
  let paws_per_dog := 4
  let nails_per_dog := nails_per_paw * paws_per_dog
  let expected_total_nails := total_dogs * nails_per_dog
  let missing_nails := expected_total_nails - total_nails
  missing_nails / nails_per_paw

theorem carly_three_legged_dogs :
  three_legged_dogs 164 11 = 3 := by
  sorry

end three_legged_dogs_carly_three_legged_dogs_l3149_314992


namespace square_difference_equality_l3149_314933

theorem square_difference_equality : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end square_difference_equality_l3149_314933


namespace quadratic_root_value_l3149_314962

theorem quadratic_root_value (a : ℝ) : 
  ((a + 1) * 1^2 - 1 + a^2 - 2*a - 2 = 0) → a = 2 :=
by
  sorry

end quadratic_root_value_l3149_314962


namespace original_number_is_fifteen_l3149_314944

theorem original_number_is_fifteen : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 105 ∧ x = 15 := by
  sorry

end original_number_is_fifteen_l3149_314944


namespace unique_solution_for_diophantine_equation_l3149_314957

theorem unique_solution_for_diophantine_equation :
  ∃! (m : ℕ+) (p q : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ 
    2^(m:ℕ) * p^2 + 1 = q^5 ∧
    m = 1 ∧ p = 11 ∧ q = 3 := by
  sorry

end unique_solution_for_diophantine_equation_l3149_314957


namespace milk_level_lowering_l3149_314927

/-- Proves that lowering the milk level by 6 inches in a 50 feet by 25 feet rectangular box removes 4687.5 gallons of milk, given that 1 cubic foot equals 7.5 gallons. -/
theorem milk_level_lowering (box_length : Real) (box_width : Real) (gallons_removed : Real) (cubic_foot_to_gallon : Real) (inches_lowered : Real) : 
  box_length = 50 ∧ 
  box_width = 25 ∧ 
  gallons_removed = 4687.5 ∧ 
  cubic_foot_to_gallon = 7.5 ∧
  inches_lowered = 6 → 
  gallons_removed = (box_length * box_width * (inches_lowered / 12)) * cubic_foot_to_gallon :=
by sorry

end milk_level_lowering_l3149_314927


namespace seating_arrangement_l3149_314924

structure Person where
  name : String
  is_sitting : Prop

def M : Person := ⟨"M", false⟩
def I : Person := ⟨"I", true⟩
def P : Person := ⟨"P", true⟩
def A : Person := ⟨"A", false⟩

theorem seating_arrangement :
  (¬M.is_sitting) →
  (¬M.is_sitting → I.is_sitting) →
  (I.is_sitting → P.is_sitting) →
  (¬A.is_sitting) →
  (I.is_sitting ∧ P.is_sitting) := by
  sorry

end seating_arrangement_l3149_314924


namespace binary_101101_conversion_l3149_314950

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, x) => acc + (if x then 2^i else 0)) 0

def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_base7 45 = [6, 3]) := by sorry

end binary_101101_conversion_l3149_314950


namespace range_of_m_for_always_nonnegative_quadratic_l3149_314964

theorem range_of_m_for_always_nonnegative_quadratic :
  {m : ℝ | ∀ x : ℝ, x^2 + m*x + 2*m + 5 ≥ 0} = {m : ℝ | -2 ≤ m ∧ m ≤ 10} := by
  sorry

end range_of_m_for_always_nonnegative_quadratic_l3149_314964


namespace prob_at_least_one_boy_and_girl_l3149_314989

-- Define the probability of a boy or girl being born
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 :=
by sorry

end prob_at_least_one_boy_and_girl_l3149_314989


namespace smallest_prime_divisor_of_sum_l3149_314929

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (3^11 + 5^13) → p = 2 :=
sorry

end smallest_prime_divisor_of_sum_l3149_314929


namespace binomial_9_choose_5_l3149_314999

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end binomial_9_choose_5_l3149_314999


namespace angle_B_is_30_degrees_l3149_314983

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the theorem
theorem angle_B_is_30_degrees (t : Triangle) 
  (h1 : t.A = 45 * π / 180)
  (h2 : t.a = 6)
  (h3 : t.b = 3 * Real.sqrt 2) :
  t.B = 30 * π / 180 :=
sorry

end angle_B_is_30_degrees_l3149_314983


namespace system_of_equations_solution_l3149_314981

theorem system_of_equations_solution :
  ∃ (x y : ℝ), x + 2*y = 3 ∧ x - 4*y = 9 → x = 5 ∧ y = -1 := by
  sorry

#check system_of_equations_solution

end system_of_equations_solution_l3149_314981


namespace min_circumcircle_area_l3149_314900

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define points A and B as tangent points on circle C
def tangent_points (xA yA xB yB : ℝ) : Prop :=
  circle_C xA yA ∧ circle_C xB yB

-- Define the theorem
theorem min_circumcircle_area (xP yP xA yA xB yB : ℝ) 
  (h_P : point_P xP yP) 
  (h_AB : tangent_points xA yA xB yB) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π = 5*π/4 ∧ 
  ∀ (r' : ℝ), r' > 0 → (∃ (xP' yP' xA' yA' xB' yB' : ℝ),
    point_P xP' yP' ∧ 
    tangent_points xA' yA' xB' yB' ∧ 
    r'^2 * π ≥ 5*π/4) :=
sorry

end min_circumcircle_area_l3149_314900


namespace tan_390_deg_l3149_314937

/-- Proves that the tangent of 390 degrees is equal to √3/3 -/
theorem tan_390_deg : Real.tan (390 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_390_deg_l3149_314937


namespace smallest_prime_sum_l3149_314977

theorem smallest_prime_sum (a b c d : ℕ) : 
  (Prime a ∧ Prime b ∧ Prime c ∧ Prime d) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  Prime (a + b + c + d) →
  (Prime (a + b) ∧ Prime (a + c) ∧ Prime (a + d) ∧ 
   Prime (b + c) ∧ Prime (b + d) ∧ Prime (c + d)) →
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ 
   Prime (a + c + d) ∧ Prime (b + c + d)) →
  a + b + c + d ≥ 31 :=
by sorry

end smallest_prime_sum_l3149_314977


namespace forty_percent_of_fifty_percent_l3149_314986

theorem forty_percent_of_fifty_percent (x : ℝ) : (0.4 * (0.5 * x)) = (0.2 * x) := by
  sorry

end forty_percent_of_fifty_percent_l3149_314986


namespace smallest_k_product_equals_sum_l3149_314960

theorem smallest_k_product_equals_sum (k : ℕ) : k = 10 ↔ 
  (k ≥ 3 ∧ 
   ∃ a b : ℕ, a ∈ Finset.range k ∧ b ∈ Finset.range k ∧ a ≠ b ∧
   a * b = (k * (k + 1) / 2) - a - b ∧
   ∀ m : ℕ, m ≥ 3 → m < k → 
     ¬∃ x y : ℕ, x ∈ Finset.range m ∧ y ∈ Finset.range m ∧ x ≠ y ∧
     x * y = (m * (m + 1) / 2) - x - y) :=
by sorry

end smallest_k_product_equals_sum_l3149_314960


namespace square_number_difference_l3149_314913

theorem square_number_difference (n k l : ℕ) :
  (∃ x : ℕ, x^2 < n ∧ n < (x+1)^2) →  -- n is between consecutive squares
  (∃ x : ℕ, n - k = x^2) →            -- n - k is a square number
  (∃ x : ℕ, n + l = x^2) →            -- n + l is a square number
  (∃ x : ℕ, n - k - l = x^2) :=        -- n - k - l is a square number
by sorry

end square_number_difference_l3149_314913


namespace direct_proportion_only_f3_l3149_314917

/-- A function f: ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- Function 1: f(x) = 3x - 4 -/
def f1 : ℝ → ℝ := λ x ↦ 3 * x - 4

/-- Function 2: f(x) = -2x + 1 -/
def f2 : ℝ → ℝ := λ x ↦ -2 * x + 1

/-- Function 3: f(x) = 3x -/
def f3 : ℝ → ℝ := λ x ↦ 3 * x

/-- Function 4: f(x) = 4 -/
def f4 : ℝ → ℝ := λ _ ↦ 4

theorem direct_proportion_only_f3 :
  ¬ is_direct_proportion f1 ∧
  ¬ is_direct_proportion f2 ∧
  is_direct_proportion f3 ∧
  ¬ is_direct_proportion f4 :=
sorry

end direct_proportion_only_f3_l3149_314917


namespace mean_of_added_numbers_l3149_314946

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 12 →
  original_list.sum / original_list.length = 75 →
  (original_list.sum + x + y + z) / (original_list.length + 3) = 90 →
  (x + y + z) / 3 = 150 := by
sorry

end mean_of_added_numbers_l3149_314946


namespace forest_leaves_count_l3149_314956

theorem forest_leaves_count :
  let trees_in_forest : ℕ := 20
  let main_branches_per_tree : ℕ := 15
  let sub_branches_per_main : ℕ := 25
  let tertiary_branches_per_sub : ℕ := 30
  let leaves_per_sub : ℕ := 75
  let leaves_per_tertiary : ℕ := 45

  let total_leaves : ℕ := 
    trees_in_forest * 
    (main_branches_per_tree * sub_branches_per_main * leaves_per_sub +
     main_branches_per_tree * sub_branches_per_main * tertiary_branches_per_sub * leaves_per_tertiary)

  total_leaves = 10687500 := by
sorry

end forest_leaves_count_l3149_314956


namespace composite_polynomial_l3149_314903

theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℕ, 1 < k ∧ k < n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1 ∧ 
  (n^(5*n-1) + n^(5*n-2) + n^(5*n-3) + n + 1) % k = 0 :=
by sorry

end composite_polynomial_l3149_314903


namespace project_time_ratio_l3149_314985

/-- Given a project where three people (Pat, Kate, and Mark) charged time, 
    prove that the ratio of time charged by Pat to Mark is 1:3 -/
theorem project_time_ratio (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 216 →
  pat_hours = 2 * kate_hours →
  mark_hours = kate_hours + 120 →
  total_hours = kate_hours + pat_hours + mark_hours →
  pat_hours * 3 = mark_hours := by
  sorry

end project_time_ratio_l3149_314985


namespace fence_pole_count_l3149_314972

/-- Represents a rectangular fence with an internal divider -/
structure RectangularFence where
  longer_side : ℕ
  shorter_side : ℕ
  has_internal_divider : Bool

/-- Calculates the total number of poles needed for a rectangular fence with an internal divider -/
def total_poles (fence : RectangularFence) : ℕ :=
  let perimeter_poles := 2 * (fence.longer_side + fence.shorter_side) - 4
  let internal_poles := if fence.has_internal_divider then fence.shorter_side - 1 else 0
  perimeter_poles + internal_poles

/-- Theorem stating that a rectangular fence with 35 poles on the longer side, 
    27 poles on the shorter side, and an internal divider needs 146 poles in total -/
theorem fence_pole_count : 
  let fence := RectangularFence.mk 35 27 true
  total_poles fence = 146 := by sorry

end fence_pole_count_l3149_314972


namespace saree_price_calculation_l3149_314984

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) = 108 → P = 150 := by
  sorry

end saree_price_calculation_l3149_314984


namespace farmers_wheat_cleaning_l3149_314982

/-- The total number of acres to be cleaned -/
def total_acres : ℕ := 480

/-- The original cleaning rate in acres per day -/
def original_rate : ℕ := 80

/-- The new cleaning rate with machinery in acres per day -/
def new_rate : ℕ := 90

/-- The number of acres cleaned on the last day -/
def last_day_acres : ℕ := 30

/-- The number of days taken to clean all acres -/
def days : ℕ := 6

theorem farmers_wheat_cleaning :
  (days - 1) * new_rate + last_day_acres = total_acres ∧
  days * original_rate = total_acres := by sorry

end farmers_wheat_cleaning_l3149_314982


namespace temperature_difference_l3149_314905

theorem temperature_difference (N : ℝ) : 
  (∃ (M L : ℝ),
    -- Noon conditions
    M = L + N ∧
    -- 6:00 PM conditions
    (M - 11) - (L + 5) = 6 ∨ (M - 11) - (L + 5) = -6) →
  N = 22 ∨ N = 10 :=
by sorry

end temperature_difference_l3149_314905


namespace largest_n_for_binomial_equality_l3149_314911

theorem largest_n_for_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6) :=
by sorry

end largest_n_for_binomial_equality_l3149_314911


namespace circle_radius_from_arc_length_l3149_314976

/-- Given a circle where the arc length corresponding to a central angle of 135° is 3π,
    prove that the radius of the circle is 4. -/
theorem circle_radius_from_arc_length :
  ∀ r : ℝ, (135 / 180 * π * r = 3 * π) → r = 4 := by
  sorry

end circle_radius_from_arc_length_l3149_314976


namespace quiz_score_theorem_l3149_314963

/-- Represents a quiz score configuration -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- The quiz scoring system -/
def quizScoring (qs : QuizScore) : ℚ :=
  4 * qs.correct + 1.5 * qs.unanswered

/-- Predicate for valid quiz configurations -/
def isValidQuizScore (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = 30

/-- Predicate for scores achievable in exactly three ways -/
def hasExactlyThreeConfigurations (score : ℚ) : Prop :=
  ∃ (c1 c2 c3 : QuizScore),
    c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    isValidQuizScore c1 ∧ isValidQuizScore c2 ∧ isValidQuizScore c3 ∧
    quizScoring c1 = score ∧ quizScoring c2 = score ∧ quizScoring c3 = score ∧
    ∀ c, isValidQuizScore c ∧ quizScoring c = score → c = c1 ∨ c = c2 ∨ c = c3

theorem quiz_score_theorem :
  ∃ score, 0 ≤ score ∧ score ≤ 120 ∧ hasExactlyThreeConfigurations score := by
  sorry

end quiz_score_theorem_l3149_314963


namespace smallest_m_is_12_l3149_314939

-- Define the set T
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

-- Define the property we want to prove
def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

-- The main theorem
theorem smallest_m_is_12 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ n ≥ m, has_nth_root_of_unity n) → m ≥ 12) :=
sorry

end smallest_m_is_12_l3149_314939


namespace scores_mode_is_80_l3149_314918

def scores : List Nat := [70, 80, 100, 60, 80, 70, 90, 50, 80, 70, 80, 70, 90, 80, 90, 80, 70, 90, 60, 80]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem scores_mode_is_80 : mode scores = some 80 := by
  sorry

end scores_mode_is_80_l3149_314918


namespace divisibility_of_2023_power_l3149_314971

theorem divisibility_of_2023_power (n : ℕ) : 
  ∃ (k : ℕ), 2023^2023 - 2023^2021 = k * 2022 * 2023 * 2024 :=
sorry

end divisibility_of_2023_power_l3149_314971


namespace implicit_function_derivative_specific_point_derivative_l3149_314907

noncomputable section

/-- The implicit function defined by 10x^3 + 4x^2y + y^2 = 0 -/
def f (x y : ℝ) : ℝ := 10 * x^3 + 4 * x^2 * y + y^2

/-- The derivative of the implicit function -/
def f_derivative (x y : ℝ) : ℝ := (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

theorem implicit_function_derivative (x y : ℝ) (h : f x y = 0) :
  deriv (fun y => f x y) y = f_derivative x y := by sorry

theorem specific_point_derivative :
  f_derivative (-2) 4 = -7/3 := by sorry

end

end implicit_function_derivative_specific_point_derivative_l3149_314907


namespace largest_two_digit_prime_factor_of_binomial_l3149_314916

theorem largest_two_digit_prime_factor_of_binomial : 
  ∃ (p : ℕ), p.Prime ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_of_binomial_l3149_314916


namespace percentage_relation_l3149_314925

theorem percentage_relation (x y z : ℝ) (hx : x = 0.06 * z) (hy : y = 0.18 * z) (hz : z > 0) :
  x / y * 100 = 100 / 3 :=
sorry

end percentage_relation_l3149_314925


namespace solution_set_equality_l3149_314928

theorem solution_set_equality (x : ℝ) : 
  Set.Icc (-1 : ℝ) (7/3 : ℝ) = { x | |x - 1| + |2*x - 1| ≤ 5 } := by
  sorry

end solution_set_equality_l3149_314928


namespace compound_statement_properties_l3149_314951

/-- Given two propositions p and q, prove the compound statement properties --/
theorem compound_statement_properties (p q : Prop) 
  (hp : p ↔ (8 + 7 = 16)) 
  (hq : q ↔ (π > 3)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p := by sorry

end compound_statement_properties_l3149_314951


namespace rotated_logarithm_function_l3149_314920

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the rotation transformation
def rotate_counterclockwise_pi_over_2 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- State the theorem
theorem rotated_logarithm_function (f : ℝ → ℝ) :
  (∀ x, rotate_counterclockwise_pi_over_2 (f x) x = (lg (x + 1), x)) →
  (∀ x, f x = 10^(-x) - 1) :=
by sorry

end rotated_logarithm_function_l3149_314920


namespace no_base_for_256_with_4_digits_l3149_314935

theorem no_base_for_256_with_4_digits :
  ¬ ∃ b : ℕ, b ≥ 2 ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_base_for_256_with_4_digits_l3149_314935


namespace total_price_houses_l3149_314996

/-- The total price of two houses, given the price of the first house and that the second house is twice as expensive. -/
def total_price (price_first_house : ℕ) : ℕ :=
  price_first_house + 2 * price_first_house

/-- Theorem stating that the total price of the two houses is $600,000 when the first house costs $200,000. -/
theorem total_price_houses : total_price 200000 = 600000 := by
  sorry

end total_price_houses_l3149_314996


namespace expression_evaluation_l3149_314994

theorem expression_evaluation (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 := by
  sorry

end expression_evaluation_l3149_314994


namespace vector_problem_l3149_314908

/-- Given two perpendicular vectors a and c, and two parallel vectors b and c in ℝ², 
    prove that x = 4, y = -8, and the magnitude of a + b is 10. -/
theorem vector_problem (x y : ℝ) 
  (a b c : ℝ × ℝ)
  (ha : a = (x, 2))
  (hb : b = (4, y))
  (hc : c = (1, -2))
  (hac_perp : a.1 * c.1 + a.2 * c.2 = 0)
  (hbc_par : b.1 * c.2 = b.2 * c.1) :
  x = 4 ∧ y = -8 ∧ Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 10 := by
  sorry

end vector_problem_l3149_314908


namespace conference_room_arrangements_count_l3149_314998

/-- The number of distinct arrangements of seats in a conference room. -/
def conference_room_arrangements : ℕ :=
  let total_seats : ℕ := 12
  let armchairs : ℕ := 6
  let benches : ℕ := 4
  let stools : ℕ := 2
  Nat.choose total_seats stools * Nat.choose (total_seats - stools) benches

theorem conference_room_arrangements_count :
  conference_room_arrangements = 13860 := by
  sorry

#eval conference_room_arrangements

end conference_room_arrangements_count_l3149_314998


namespace tangent_line_b_value_l3149_314959

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at the point (2, 3), prove that b = -15 -/
theorem tangent_line_b_value (k a : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2, 3)
  (3 = 8 + 2 * a + 1) →  -- Curve equation at (2, 3)
  (k = 3 * 2^2 + a) →  -- Slope of the tangent line equals derivative of the curve at x = 2
  (b = -15) := by sorry

end tangent_line_b_value_l3149_314959


namespace original_deck_size_is_52_l3149_314990

/-- The number of players among whom the deck is distributed -/
def num_players : ℕ := 3

/-- The number of cards each player receives after distribution -/
def cards_per_player : ℕ := 18

/-- The number of cards added to the original deck -/
def added_cards : ℕ := 2

/-- The original number of cards in the deck -/
def original_deck_size : ℕ := num_players * cards_per_player - added_cards

theorem original_deck_size_is_52 : original_deck_size = 52 := by
  sorry

end original_deck_size_is_52_l3149_314990


namespace M_sufficient_not_necessary_for_N_l3149_314967

def M : Set ℝ := {x | x^2 < 3*x}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_sufficient_not_necessary_for_N :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ (∃ b : ℝ, b ∈ N ∧ b ∉ M) := by sorry

end M_sufficient_not_necessary_for_N_l3149_314967


namespace students_without_A_l3149_314993

/-- Given a class of students and their grades in three subjects, calculate the number of students who didn't receive an A in any subject. -/
theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ) 
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) : 
  total = 40 →
  history = 10 →
  math = 15 →
  science = 8 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 17 := by
sorry

end students_without_A_l3149_314993


namespace xyz_sum_root_l3149_314949

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 22 / 2)
  (h2 : z + x = 24 / 2)
  (h3 : x + y = 26 / 2) :
  Real.sqrt (x * y * z * (x + y + z)) = 3 * Real.sqrt 70 := by
  sorry

end xyz_sum_root_l3149_314949


namespace simplify_expression_l3149_314958

theorem simplify_expression (w : ℝ) : 
  2 * w + 3 - 4 * w - 6 + 7 * w + 9 - 8 * w - 12 + 3 * (2 * w - 1) = 3 * w - 9 := by
  sorry

end simplify_expression_l3149_314958


namespace saras_team_games_l3149_314940

/-- The total number of games played by Sara's high school basketball team -/
def total_games (won_games defeated_games : ℕ) : ℕ :=
  won_games + defeated_games

/-- Theorem stating that for Sara's team, the total number of games
    is equal to the sum of won games and defeated games -/
theorem saras_team_games :
  total_games 12 4 = 16 := by
  sorry

end saras_team_games_l3149_314940


namespace max_value_inequality_l3149_314969

theorem max_value_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  x * y * z * (x + y + z + w) / ((x + y + z)^2 * (y + z + w)^2) ≤ 1/4 := by
  sorry

end max_value_inequality_l3149_314969


namespace negative_inequality_l3149_314922

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end negative_inequality_l3149_314922


namespace inequality_solution_l3149_314995

theorem inequality_solution (x : ℝ) : 
  (202 * Real.sqrt (x^3 - 2*x - 2/x + 1/x^3 + 4) ≤ 0) ↔ 
  (x = (-1 - Real.sqrt 17 + Real.sqrt (2 * Real.sqrt 17 + 2)) / 4 ∨ 
   x = (-1 - Real.sqrt 17 - Real.sqrt (2 * Real.sqrt 17 + 2)) / 4) :=
by sorry

end inequality_solution_l3149_314995


namespace negation_constant_arithmetic_sequence_l3149_314904

theorem negation_constant_arithmetic_sequence :
  ¬(∀ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) → (∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) ↔
  (∃ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) ∧ ¬(∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) :=
by sorry

end negation_constant_arithmetic_sequence_l3149_314904


namespace cl2_moles_required_l3149_314978

-- Define the reaction components
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2cl6 : ℝ
  hcl : ℝ

-- Define the balanced equation ratios
def balancedRatio : Reaction := {
  c2h6 := 1,
  cl2 := 6,
  c2cl6 := 1,
  hcl := 6
}

-- Define the given reaction
def givenReaction : Reaction := {
  c2h6 := 2,
  cl2 := 0,  -- This is what we need to prove
  c2cl6 := 2,
  hcl := 12
}

-- Theorem statement
theorem cl2_moles_required (r : Reaction) :
  r.c2h6 = givenReaction.c2h6 ∧
  r.c2cl6 = givenReaction.c2cl6 ∧
  r.hcl = givenReaction.hcl →
  r.cl2 = 12 :=
by sorry

end cl2_moles_required_l3149_314978


namespace conference_seating_arrangement_l3149_314966

theorem conference_seating_arrangement :
  ∃! y : ℕ, ∃ x : ℕ,
    (9 * x + 10 * y = 73) ∧
    (0 < x) ∧ (0 < y) ∧
    y = 1 := by
  sorry

end conference_seating_arrangement_l3149_314966
