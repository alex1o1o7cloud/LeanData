import Mathlib

namespace viewing_angle_midpoint_l2230_223020

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the viewing angle function
noncomputable def viewingAngle (c : Circle) (p : Point) : ℝ := sorry

-- Define the line AB
def lineAB (A B : Point) : Set Point := sorry

-- Theorem statement
theorem viewing_angle_midpoint (O : Circle) (A B : Point) :
  let α := viewingAngle O A
  let β := viewingAngle O B
  let γ := (α + β) / 2
  ∃ (C₁ C₂ : Point), C₁ ∈ lineAB A B ∧ C₂ ∈ lineAB A B ∧
    viewingAngle O C₁ = γ ∧ viewingAngle O C₂ = γ ∧
    (α = β → (C₁ = A ∧ C₂ = B) ∨ (C₁ = B ∧ C₂ = A)) :=
by sorry


end viewing_angle_midpoint_l2230_223020


namespace shoe_price_calculation_l2230_223087

theorem shoe_price_calculation (thursday_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  thursday_price = 50 →
  friday_increase = 0.2 →
  monday_decrease = 0.15 →
  thursday_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry


end shoe_price_calculation_l2230_223087


namespace first_three_decimal_digits_l2230_223059

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → x = (10^n + 1)^(11/7) → 
  ∃ (y : ℝ), x = 10^2861 + y ∧ 0.571 < y/10^858 ∧ y/10^858 < 0.572 :=
by sorry

end first_three_decimal_digits_l2230_223059


namespace coin_flip_problem_l2230_223042

theorem coin_flip_problem : ∃ (n : ℕ+) (a b : ℕ),
  a + b = n ∧
  4 + 8 * a - 3 * b = 1 + 3 * 2^(a - b) ∧
  (4 + 8 * a - 3 * b : ℤ) < 2012 ∧
  n = 137 := by
sorry

end coin_flip_problem_l2230_223042


namespace remainder_proof_l2230_223037

theorem remainder_proof : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end remainder_proof_l2230_223037


namespace surface_area_combined_shape_l2230_223093

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified shape -/
def surfaceAreaModifiedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Calculates the surface area of the combined shape -/
def surfaceAreaCombinedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the combined shape is 38 cm² -/
theorem surface_area_combined_shape :
  let original := CubeDimensions.mk 2 2 2
  let removed := CubeDimensions.mk 1 1 1
  surfaceAreaCombinedShape original removed = 38 := by
  sorry

end surface_area_combined_shape_l2230_223093


namespace cookie_pans_problem_l2230_223021

/-- Given a number of cookies per pan and a total number of cookies,
    calculate the number of pans needed. -/
def calculate_pans (cookies_per_pan : ℕ) (total_cookies : ℕ) : ℕ :=
  total_cookies / cookies_per_pan

theorem cookie_pans_problem :
  let cookies_per_pan : ℕ := 8
  let total_cookies : ℕ := 40
  calculate_pans cookies_per_pan total_cookies = 5 := by
  sorry

#eval calculate_pans 8 40

end cookie_pans_problem_l2230_223021


namespace min_occupied_seats_l2230_223060

/-- Represents the seating arrangement problem --/
def SeatingArrangement (total_seats : ℕ) (pattern : List ℕ) (occupied : ℕ) : Prop :=
  -- The total number of seats is 150
  total_seats = 150 ∧
  -- The pattern alternates between 4 and 3 empty seats
  pattern = [4, 3] ∧
  -- The occupied seats ensure the next person must sit next to someone
  occupied ≥ 
    -- Calculate the minimum number of occupied seats
    let full_units := total_seats / (pattern.sum + pattern.length)
    let remaining_seats := total_seats % (pattern.sum + pattern.length)
    let seats_in_full_units := full_units * pattern.length
    let additional_seats := if remaining_seats ≥ pattern.head! then 2 else 0
    seats_in_full_units + additional_seats

/-- The theorem stating the minimum number of occupied seats --/
theorem min_occupied_seats :
  ∃ (occupied : ℕ), SeatingArrangement 150 [4, 3] occupied ∧ occupied = 50 := by
  sorry

end min_occupied_seats_l2230_223060


namespace isosceles_in_26gon_l2230_223082

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- Main theorem: In a regular 26-gon, any 9 vertices contain an isosceles triangle -/
theorem isosceles_in_26gon (p : RegularPolygon 26) 
  (vertices : Finset (Fin 26)) (h : vertices.card = 9) :
  ∃ (v1 v2 v3 : Fin 26), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v3 ∈ vertices ∧
    IsIsoscelesTriangle p v1 v2 v3 := by
  sorry

end isosceles_in_26gon_l2230_223082


namespace tangent_slope_and_extrema_l2230_223095

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem tangent_slope_and_extrema (a : ℝ) :
  (deriv (f a) 0 = 1) →
  (a = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 0 ≤ f 1 x) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 x ≤ f 1 2) ∧
  (f 1 0 = 0) ∧
  (f 1 2 = 2 * Real.exp 2) :=
by sorry

end

end tangent_slope_and_extrema_l2230_223095


namespace cos_alpha_value_l2230_223024

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end cos_alpha_value_l2230_223024


namespace distinct_integers_with_divisibility_property_l2230_223098

theorem distinct_integers_with_divisibility_property (n : ℕ) (h : n ≥ 2) :
  ∃ (a : Fin n → ℕ+), (∀ i j, i.val < j.val → (a i).val ≠ (a j).val) ∧
    (∀ i j, i.val < j.val → ((a i).val - (a j).val) ∣ ((a i).val + (a j).val)) := by
  sorry

end distinct_integers_with_divisibility_property_l2230_223098


namespace square_area_increase_l2230_223092

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
  sorry

end square_area_increase_l2230_223092


namespace proportion_solution_l2230_223079

theorem proportion_solution (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 4 ∧ x = 0.39999999999999997 → y = 6 := by
  sorry

end proportion_solution_l2230_223079


namespace second_class_size_l2230_223032

theorem second_class_size 
  (first_class_size : ℕ) 
  (first_class_avg : ℚ) 
  (second_class_avg : ℚ) 
  (total_avg : ℚ) 
  (h1 : first_class_size = 35)
  (h2 : first_class_avg = 40)
  (h3 : second_class_avg = 60)
  (h4 : total_avg = 51.25) :
  ∃ second_class_size : ℕ,
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 45 := by
  sorry


end second_class_size_l2230_223032


namespace lena_kevin_ratio_l2230_223072

-- Define the initial number of candy bars for Lena
def lena_initial : ℕ := 16

-- Define the number of additional candy bars Lena needs
def additional_candies : ℕ := 5

-- Define the relationship between Lena's and Nicole's candy bars
def lena_nicole_diff : ℕ := 5

-- Define the relationship between Nicole's and Kevin's candy bars
def nicole_kevin_diff : ℕ := 4

-- Calculate Nicole's candy bars
def nicole_candies : ℕ := lena_initial - lena_nicole_diff

-- Calculate Kevin's candy bars
def kevin_candies : ℕ := nicole_candies - nicole_kevin_diff

-- Calculate Lena's final number of candy bars
def lena_final : ℕ := lena_initial + additional_candies

-- Theorem stating the ratio of Lena's final candy bars to Kevin's candy bars
theorem lena_kevin_ratio : 
  lena_final / kevin_candies = 3 ∧ lena_final % kevin_candies = 0 := by
  sorry

end lena_kevin_ratio_l2230_223072


namespace units_digit_of_expression_l2230_223069

theorem units_digit_of_expression (k : ℕ) (h : k = 2012^2 + 2^2012) :
  (k^3 + 2^(k+1)) % 10 = 2 := by
  sorry

end units_digit_of_expression_l2230_223069


namespace inequality_range_l2230_223066

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x + 1 > 2*x + m) → m < -1 :=
by sorry

end inequality_range_l2230_223066


namespace triangle_side_length_l2230_223010

theorem triangle_side_length (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →  -- positive side lengths
  (|a - 7| + (b - 2)^2 = 0) →    -- given equation
  (∃ n : ℕ, c = 2*n + 1) →      -- c is odd
  (a + b > c) →                 -- triangle inequality
  (a + c > b) →                 -- triangle inequality
  (b + c > a) →                 -- triangle inequality
  (c = 7) :=                    -- conclusion
by
  sorry

end triangle_side_length_l2230_223010


namespace five_long_sides_l2230_223041

/-- A convex hexagon with specific properties -/
structure ConvexHexagon where
  -- The two distinct side lengths
  short_side : ℝ
  long_side : ℝ
  -- The number of sides with each length
  num_short_sides : ℕ
  num_long_sides : ℕ
  -- Properties
  is_convex : Bool
  distinct_lengths : short_side ≠ long_side
  total_sides : num_short_sides + num_long_sides = 6
  perimeter : num_short_sides * short_side + num_long_sides * long_side = 40
  short_side_length : short_side = 4
  long_side_length : long_side = 7

/-- Theorem: In a convex hexagon with the given properties, there are exactly 5 sides measuring 7 units -/
theorem five_long_sides (h : ConvexHexagon) : h.num_long_sides = 5 :=
  sorry

end five_long_sides_l2230_223041


namespace tenth_configuration_stones_l2230_223054

/-- The number of stones in the n-th configuration of Anya's pentagon pattern -/
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define n = 0 as having no stones for completeness
  | 1 => 1
  | n + 1 => stones n + 3 * (n + 1) - 2

/-- The theorem stating that the 10th configuration has 145 stones -/
theorem tenth_configuration_stones :
  stones 10 = 145 := by
  sorry

/-- Helper lemma to show the first four configurations match the given values -/
lemma first_four_configurations :
  stones 1 = 1 ∧ stones 2 = 5 ∧ stones 3 = 12 ∧ stones 4 = 22 := by
  sorry

end tenth_configuration_stones_l2230_223054


namespace polynomial_b_value_l2230_223083

theorem polynomial_b_value (A B : ℤ) : 
  let p := fun z : ℝ => z^4 - 9*z^3 + A*z^2 + B*z + 18
  (∃ r1 r2 r3 r4 : ℕ+, (p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0 ∧ p r4 = 0) ∧ 
                       (r1 + r2 + r3 + r4 = 9)) →
  B = -20 := by
sorry

end polynomial_b_value_l2230_223083


namespace circle_translation_sum_l2230_223075

/-- The equation of circle D before translation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 10*y = -14

/-- The center of the circle after translation -/
def new_center : ℝ × ℝ := (1, -2)

/-- Theorem stating the sum of new center coordinates and radius after translation -/
theorem circle_translation_sum :
  ∃ (r : ℝ), 
    (∀ x y : ℝ, circle_equation x y → 
      ∃ a b : ℝ, new_center = (a, b) ∧ 
        a + b + r = -1 + Real.sqrt 27) :=
sorry

end circle_translation_sum_l2230_223075


namespace same_color_probability_l2230_223047

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end same_color_probability_l2230_223047


namespace cookie_ratio_l2230_223076

/-- Represents the cookie distribution problem --/
def cookie_problem (initial_cookies : ℕ) (given_to_brother : ℕ) (left_at_end : ℕ) : Prop :=
  let mother_gift := given_to_brother / 2
  let total_after_mother := initial_cookies - given_to_brother + mother_gift
  let given_to_sister := total_after_mother - left_at_end
  (given_to_sister : ℚ) / total_after_mother = 2 / 3

/-- The main theorem stating the cookie distribution ratio --/
theorem cookie_ratio : 
  cookie_problem 20 10 5 := by sorry

end cookie_ratio_l2230_223076


namespace inequality_proof_l2230_223046

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by sorry

end inequality_proof_l2230_223046


namespace exists_valid_coloring_l2230_223078

/-- A coloring of a complete graph with 6 vertices using 5 colors -/
def GraphColoring : Type := Fin 6 → Fin 6 → Fin 5

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (c : GraphColoring) : Prop :=
  ∀ v : Fin 6, ∀ u w : Fin 6, u ≠ v → w ≠ v → u ≠ w → c v u ≠ c v w

/-- Theorem stating that a valid 5-coloring exists for a complete graph with 6 vertices -/
theorem exists_valid_coloring : ∃ c : GraphColoring, is_valid_coloring c := by
  sorry

end exists_valid_coloring_l2230_223078


namespace arrangements_without_A_at_head_l2230_223050

def total_people : Nat := 5
def people_to_select : Nat := 3

def total_arrangements : Nat := total_people * (total_people - 1) * (total_people - 2)
def arrangements_with_A_at_head : Nat := (total_people - 1) * (total_people - 2)

theorem arrangements_without_A_at_head :
  total_arrangements - arrangements_with_A_at_head = 36 := by
  sorry

end arrangements_without_A_at_head_l2230_223050


namespace bryce_raisins_l2230_223017

/-- Proves that Bryce received 12 raisins given the conditions of the problem -/
theorem bryce_raisins : 
  ∀ (bryce carter emma : ℕ), 
    bryce = carter + 8 →
    carter = bryce / 3 →
    emma = 2 * carter →
    bryce = 12 := by
  sorry

end bryce_raisins_l2230_223017


namespace pure_imaginary_ratio_l2230_223009

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4*I) * (a + b*I) = y*I) : a/b = -4/3 := by
  sorry

end pure_imaginary_ratio_l2230_223009


namespace factorization_of_x_squared_plus_2x_l2230_223061

theorem factorization_of_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end factorization_of_x_squared_plus_2x_l2230_223061


namespace exist_three_similar_non_congruent_triangles_l2230_223003

/-- A structure representing a triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle_B : ℝ)

/-- Definition of similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.angle_B = t2.angle_B

/-- Definition of congruence between two triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧ t1.angle_B = t2.angle_B

/-- Theorem stating the existence of three pairwise similar but non-congruent triangles -/
theorem exist_three_similar_non_congruent_triangles :
  ∃ (t1 t2 t3 : Triangle),
    similar t1 t2 ∧ similar t2 t3 ∧ similar t3 t1 ∧
    ¬congruent t1 t2 ∧ ¬congruent t2 t3 ∧ ¬congruent t3 t1 :=
by
  sorry

end exist_three_similar_non_congruent_triangles_l2230_223003


namespace root_existence_implies_m_range_l2230_223071

theorem root_existence_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0) → -3 ≤ m ∧ m < 0 := by
  sorry

end root_existence_implies_m_range_l2230_223071


namespace brandy_trail_mix_chocolate_chips_l2230_223096

/-- The weight of chocolate chips in Brandy's trail mix -/
def weight_chocolate_chips (total_weight peanuts_weight raisins_weight : ℚ) : ℚ :=
  total_weight - (peanuts_weight + raisins_weight)

/-- Theorem stating that the weight of chocolate chips in Brandy's trail mix is 0.17 pounds -/
theorem brandy_trail_mix_chocolate_chips :
  weight_chocolate_chips 0.42 0.17 0.08 = 0.17 := by
  sorry

end brandy_trail_mix_chocolate_chips_l2230_223096


namespace F_opposite_A_l2230_223038

/-- Represents a face of a cube --/
inductive CubeFace
| A | B | C | D | E | F

/-- Represents the position of a face relative to face A in the net --/
inductive Position
| Left | Above | Right | Below | NotAttached

/-- Describes the layout of faces in the cube net --/
def net_layout : CubeFace → Position
| CubeFace.B => Position.Left
| CubeFace.C => Position.Above
| CubeFace.D => Position.Right
| CubeFace.E => Position.Below
| CubeFace.F => Position.NotAttached
| CubeFace.A => Position.NotAttached  -- A's position relative to itself is not relevant

/-- Determines if two faces are opposite in the folded cube --/
def are_opposite (f1 f2 : CubeFace) : Prop := sorry

/-- Theorem stating that face F is opposite to face A when the net is folded --/
theorem F_opposite_A : are_opposite CubeFace.F CubeFace.A := by
  sorry

end F_opposite_A_l2230_223038


namespace num_chords_ten_points_l2230_223028

/-- The number of chords formed by connecting 2 points out of n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 10 points marked on the circumference of a circle -/
def num_points : ℕ := 10

/-- Theorem: The number of chords formed by connecting 2 points out of 10 points on a circle is 45 -/
theorem num_chords_ten_points : num_chords num_points = 45 := by
  sorry

end num_chords_ten_points_l2230_223028


namespace daily_wage_calculation_l2230_223049

def days_in_week : ℕ := 7
def weeks : ℕ := 6
def total_earnings : ℕ := 2646

theorem daily_wage_calculation (days_worked : ℕ) (daily_wage : ℚ) 
  (h1 : days_worked = days_in_week * weeks)
  (h2 : daily_wage * days_worked = total_earnings) :
  daily_wage = 63 := by sorry

end daily_wage_calculation_l2230_223049


namespace algebraic_simplification_l2230_223048

theorem algebraic_simplification (a b : ℝ) :
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b := by
  sorry

end algebraic_simplification_l2230_223048


namespace motorcycle_journey_avg_speed_l2230_223097

/-- A motorcyclist's journey with specific conditions -/
def motorcycle_journey (distance_AB : ℝ) (speed_BC : ℝ) : Prop :=
  ∃ (time_AB time_BC : ℝ),
    time_AB > 0 ∧ time_BC > 0 ∧
    time_AB = 3 * time_BC ∧
    distance_AB = 120 ∧
    speed_BC = 60 ∧
    (distance_AB / 2) / time_BC = speed_BC ∧
    (distance_AB + distance_AB / 2) / (time_AB + time_BC) = 45

/-- Theorem stating that under the given conditions, the average speed is 45 mph -/
theorem motorcycle_journey_avg_speed :
  motorcycle_journey 120 60 :=
sorry

end motorcycle_journey_avg_speed_l2230_223097


namespace podcast_ratio_l2230_223035

def total_drive_time : ℕ := 360 -- in minutes
def first_podcast : ℕ := 45 -- in minutes
def third_podcast : ℕ := 105 -- in minutes
def fourth_podcast : ℕ := 60 -- in minutes
def next_podcast : ℕ := 60 -- in minutes

theorem podcast_ratio : 
  let second_podcast := total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast)
  (second_podcast : ℚ) / first_podcast = 2 := by
sorry

end podcast_ratio_l2230_223035


namespace proposition_relation_l2230_223051

theorem proposition_relation (a b : ℝ) : 
  (∃ a b : ℝ, |a - b| < 3 ∧ (|a| ≥ 1 ∨ |b| ≥ 2)) ∧
  (∀ a b : ℝ, |a| < 1 ∧ |b| < 2 → |a - b| < 3) :=
by sorry

end proposition_relation_l2230_223051


namespace vector_subtraction_l2230_223005

/-- Given two complex numbers representing vectors OA and OB, 
    prove that the complex number representing vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℂ) : 
  OA = 5 + 10*I → OB = 3 - 4*I → (OB - OA) = -2 - 14*I := by
  sorry

end vector_subtraction_l2230_223005


namespace complex_number_in_fourth_quadrant_l2230_223012

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := 3 - Complex.I * (a^2 + 1)
  z.re > 0 ∧ z.im < 0 :=
by sorry

end complex_number_in_fourth_quadrant_l2230_223012


namespace range_of_a_l2230_223068

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 5 → x^2 - 2*x + a ≥ 0) ↔ a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l2230_223068


namespace ratio_problem_l2230_223094

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : 
  x / y = 11 / 6 := by
  sorry

end ratio_problem_l2230_223094


namespace yoojung_initial_candies_l2230_223039

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left over -/
def candies_left_over : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left_over

theorem yoojung_initial_candies : initial_candies = 28 := by
  sorry

end yoojung_initial_candies_l2230_223039


namespace doll_collection_increase_l2230_223045

theorem doll_collection_increase (original_count : ℕ) (increase_percentage : ℚ) (final_count : ℕ) 
  (h1 : increase_percentage = 25 / 100)
  (h2 : final_count = 10)
  (h3 : final_count = original_count + (increase_percentage * original_count).floor) :
  final_count - original_count = 2 := by
  sorry

end doll_collection_increase_l2230_223045


namespace river_bank_bottom_width_l2230_223008

/-- Given a trapezium-shaped cross-section of a river bank, prove that the bottom width is 8 meters -/
theorem river_bank_bottom_width
  (top_width : ℝ)
  (depth : ℝ)
  (area : ℝ)
  (h1 : top_width = 12)
  (h2 : depth = 50)
  (h3 : area = 500)
  (h4 : area = (top_width + bottom_width) * depth / 2) :
  bottom_width = 8 :=
by
  sorry

end river_bank_bottom_width_l2230_223008


namespace yang_hui_field_theorem_l2230_223019

/-- Represents a rectangular field with given area and perimeter --/
structure RectangularField where
  area : ℕ
  perimeter : ℕ

/-- Calculates the difference between length and width of a rectangular field --/
def lengthWidthDifference (field : RectangularField) : ℕ :=
  let length := (field.perimeter + (field.perimeter^2 - 16 * field.area).sqrt) / 4
  let width := field.perimeter / 2 - length
  length - width

/-- Theorem stating the difference between length and width for the specific field --/
theorem yang_hui_field_theorem : 
  ∀ (field : RectangularField), 
  field.area = 864 ∧ field.perimeter = 120 → lengthWidthDifference field = 12 := by
  sorry

end yang_hui_field_theorem_l2230_223019


namespace exam_pass_count_l2230_223018

theorem exam_pass_count (total : ℕ) (overall_avg passed_avg failed_avg : ℚ) : 
  total = 120 →
  overall_avg = 35 →
  passed_avg = 39 →
  failed_avg = 15 →
  ∃ (passed : ℕ), 
    passed ≤ total ∧ 
    (passed : ℚ) * passed_avg + (total - passed : ℚ) * failed_avg = (total : ℚ) * overall_avg ∧
    passed = 100 := by
  sorry

end exam_pass_count_l2230_223018


namespace complex_addition_l2230_223011

theorem complex_addition : (1 : ℂ) + 3*I + (2 : ℂ) - 4*I = 3 - I := by sorry

end complex_addition_l2230_223011


namespace product_properties_l2230_223002

-- Define a function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

theorem product_properties :
  (trailingZeros (360 * 5) = 2) ∧ (250 * 4 = 1000) := by
  sorry

end product_properties_l2230_223002


namespace cards_added_l2230_223004

theorem cards_added (initial_cards : ℕ) (total_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 4 → total_cards = 7 → total_cards = initial_cards + added_cards → added_cards = 3 := by
  sorry

end cards_added_l2230_223004


namespace alternating_color_probability_l2230_223022

def box := {white : ℕ // white = 5} × {black : ℕ // black = 5}

def total_arrangements (b : box) : ℕ := Nat.choose (b.1 + b.2) b.1

def alternating_arrangements : ℕ := 2

theorem alternating_color_probability (b : box) :
  (alternating_arrangements : ℚ) / (total_arrangements b : ℚ) = 1 / 126 :=
sorry

end alternating_color_probability_l2230_223022


namespace largest_solution_of_equation_l2230_223084

theorem largest_solution_of_equation (x : ℝ) :
  (x / 5 + 1 / (5 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end largest_solution_of_equation_l2230_223084


namespace existence_of_differences_l2230_223062

theorem existence_of_differences (n : ℕ) (x : Fin n → Fin n → ℚ)
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℚ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end existence_of_differences_l2230_223062


namespace square_root_of_four_l2230_223064

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l2230_223064


namespace perfect_square_base8_l2230_223040

/-- Represents a number in base 8 of the form ab3c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 24 + n.c

theorem perfect_square_base8 (n : Base8Number) :
  (∃ m : Nat, toDecimal n = m * m) → n.c = 1 := by
  sorry

end perfect_square_base8_l2230_223040


namespace base_5_to_binary_44_l2230_223029

def base_5_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 5 + (n % 10)

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem base_5_to_binary_44 :
  decimal_to_binary (base_5_to_decimal 44) = [1, 1, 0, 0, 0] := by
  sorry

end base_5_to_binary_44_l2230_223029


namespace function_identity_l2230_223077

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
sorry

end function_identity_l2230_223077


namespace amanda_remaining_money_l2230_223088

/-- Calculates the remaining money after Amanda's purchases -/
def remaining_money (gift_amount : ℚ) (tape_price : ℚ) (num_tapes : ℕ) 
  (headphone_price : ℚ) (vinyl_price : ℚ) (poster_price : ℚ) 
  (tape_discount : ℚ) (headphone_tax : ℚ) (shipping_cost : ℚ) : ℚ :=
  let tape_total := tape_price * num_tapes * (1 - tape_discount)
  let headphone_total := headphone_price * (1 + headphone_tax)
  let total_cost := tape_total + headphone_total + vinyl_price + poster_price + shipping_cost
  gift_amount - total_cost

/-- Theorem stating that Amanda will have $16.75 left after her purchases -/
theorem amanda_remaining_money :
  remaining_money 200 15 3 55 35 45 0.1 0.05 5 = 16.75 := by
  sorry

end amanda_remaining_money_l2230_223088


namespace perfect_square_sum_permutation_l2230_223085

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def valid_permutation (n : ℕ) (p : Fin n → Fin n) : Prop :=
  Function.Bijective p ∧ ∀ i : Fin n, is_perfect_square ((i.val + 1) + (p i).val + 1)

theorem perfect_square_sum_permutation :
  (∃ p : Fin 9 → Fin 9, valid_permutation 9 p) ∧
  (¬ ∃ p : Fin 11 → Fin 11, valid_permutation 11 p) ∧
  (∃ p : Fin 1996 → Fin 1996, valid_permutation 1996 p) := by
  sorry

end perfect_square_sum_permutation_l2230_223085


namespace indeterminate_equation_solutions_l2230_223043

def solution_set : Set (ℤ × ℤ) := {(3, -1), (5, 1), (1, 5), (-1, 3)}

theorem indeterminate_equation_solutions :
  {(x, y) : ℤ × ℤ | 2 * (x + y) = x * y + 7} = solution_set := by
  sorry

end indeterminate_equation_solutions_l2230_223043


namespace cuboid_height_l2230_223014

theorem cuboid_height (volume : ℝ) (base_area : ℝ) (height : ℝ) 
  (h1 : volume = 144)
  (h2 : base_area = 18)
  (h3 : volume = base_area * height) :
  height = 8 := by
sorry

end cuboid_height_l2230_223014


namespace quadratic_two_roots_range_l2230_223015

theorem quadratic_two_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1/4 = 0 ∧ y^2 + m*y + 1/4 = 0) ↔ 
  (m < -1 ∨ m > 1) :=
sorry

end quadratic_two_roots_range_l2230_223015


namespace cookie_sugar_measurement_l2230_223053

def sugar_needed : ℚ := 15/4  -- 3¾ cups of sugar
def cup_capacity : ℚ := 1/3   -- ⅓ cup measuring cup

theorem cookie_sugar_measurement : ∃ n : ℕ, n * cup_capacity ≥ sugar_needed ∧ 
  ∀ m : ℕ, m * cup_capacity ≥ sugar_needed → n ≤ m :=
by
  sorry

end cookie_sugar_measurement_l2230_223053


namespace fifth_closest_is_park_l2230_223081

def buildings := ["bank", "school", "stationery store", "convenience store", "park"]

theorem fifth_closest_is_park :
  buildings.get? 4 = some "park" :=
sorry

end fifth_closest_is_park_l2230_223081


namespace ralph_squares_count_l2230_223031

/-- The number of matchsticks in a box -/
def total_matchsticks : ℕ := 50

/-- The number of matchsticks Elvis uses for one square -/
def elvis_square_size : ℕ := 4

/-- The number of matchsticks Ralph uses for one square -/
def ralph_square_size : ℕ := 8

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

theorem ralph_squares_count : 
  elvis_square_size * elvis_squares + ralph_square_size * ralph_squares + remaining_matchsticks = total_matchsticks :=
by sorry

end ralph_squares_count_l2230_223031


namespace even_quadratic_implies_b_zero_l2230_223070

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem even_quadratic_implies_b_zero (b c : ℝ) :
  IsEven (f b c) → b = 0 := by
  sorry

end even_quadratic_implies_b_zero_l2230_223070


namespace coal_supply_duration_l2230_223034

/-- 
Given a coal supply that was originally planned to last for a certain number of days 
with a specific daily consumption, and the actual daily consumption being a percentage 
less than planned, calculate the number of days the coal supply will actually last.
-/
theorem coal_supply_duration 
  (planned_daily_consumption : ℝ) 
  (planned_duration : ℝ) 
  (consumption_reduction_percentage : ℝ) : 
  planned_daily_consumption = 0.25 →
  planned_duration = 80 →
  consumption_reduction_percentage = 20 →
  (planned_daily_consumption * planned_duration) / 
  (planned_daily_consumption * (1 - consumption_reduction_percentage / 100)) = 100 := by
  sorry

end coal_supply_duration_l2230_223034


namespace candy_game_solution_l2230_223023

/-- 
Given a game where:
- 50 questions are asked
- Correct answers result in gaining 7 candies
- Incorrect answers result in losing 3 candies
- The net change in candies is zero

Prove that the number of correctly answered questions is 15.
-/
theorem candy_game_solution (total_questions : Nat) 
  (correct_reward : Nat) (incorrect_penalty : Nat) 
  (x : Nat) : 
  total_questions = 50 → 
  correct_reward = 7 → 
  incorrect_penalty = 3 → 
  x * correct_reward = (total_questions - x) * incorrect_penalty → 
  x = 15 := by
  sorry

end candy_game_solution_l2230_223023


namespace raffle_winnings_l2230_223067

theorem raffle_winnings (W : ℝ) (h1 : W > 0) (h2 : W / 2 - 2 + 114 = W) : 
  W - W / 2 - 2 = 110 := by
  sorry

end raffle_winnings_l2230_223067


namespace same_color_plate_probability_l2230_223080

theorem same_color_plate_probability :
  let total_plates : ℕ := 12
  let red_plates : ℕ := 7
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  (red_combinations + blue_combinations : ℚ) / total_combinations = 9 / 44 :=
by sorry

end same_color_plate_probability_l2230_223080


namespace unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l2230_223052

theorem unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3 :
  ∃! n : ℕ+, 20 ∣ n ∧ (8.2 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 8.3 :=
by
  -- The proof goes here
  sorry

end unique_integer_divisible_by_20_cube_root_between_8_2_and_8_3_l2230_223052


namespace quadratic_function_minimum_l2230_223057

theorem quadratic_function_minimum (a b c : ℝ) 
  (h1 : b > a) 
  (h2 : a > 0) 
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) : 
  (a + b + c) / (b - a) ≥ 3 := by sorry

end quadratic_function_minimum_l2230_223057


namespace cone_lateral_surface_area_l2230_223074

/-- Represents a cone with given slant height and lateral surface property -/
structure Cone where
  slant_height : ℝ
  lateral_surface_is_semicircle : Prop

/-- Calculates the lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

theorem cone_lateral_surface_area 
  (c : Cone) 
  (h1 : c.slant_height = 10) 
  (h2 : c.lateral_surface_is_semicircle) : 
  lateral_surface_area c = 50 * Real.pi := by sorry

end cone_lateral_surface_area_l2230_223074


namespace final_walnuts_count_l2230_223026

-- Define the initial conditions and actions
def initial_walnuts : ℕ := 25
def boy_gathered : ℕ := 15
def boy_dropped : ℕ := 3
def boy_hidden : ℕ := 5
def girl_brought : ℕ := 12
def girl_eaten : ℕ := 4
def girl_given : ℕ := 3
def girl_lost : ℕ := 2

-- Theorem to prove
theorem final_walnuts_count :
  initial_walnuts + 
  (boy_gathered - boy_dropped - boy_hidden) + 
  (girl_brought - girl_eaten - girl_given - girl_lost) = 35 := by
  sorry

end final_walnuts_count_l2230_223026


namespace probability_of_C_l2230_223073

-- Define the wheel with four parts
inductive WheelPart : Type
| A
| B
| C
| D

-- Define the probability function
def probability : WheelPart → ℚ
| WheelPart.A => 1/4
| WheelPart.B => 1/3
| WheelPart.C => 1/4  -- This is what we want to prove
| WheelPart.D => 1/6

-- State the theorem
theorem probability_of_C : probability WheelPart.C = 1/4 := by
  -- The sum of all probabilities must equal 1
  have sum_of_probabilities : 
    probability WheelPart.A + probability WheelPart.B + 
    probability WheelPart.C + probability WheelPart.D = 1 := by sorry
  
  -- Proof goes here
  sorry

end probability_of_C_l2230_223073


namespace fraction_calculation_l2230_223036

theorem fraction_calculation : 
  (1/5 - 1/3) / ((3/7) / (2/9)) = -28/405 := by
  sorry

end fraction_calculation_l2230_223036


namespace linear_function_properties_l2230_223089

def LinearFunction (m c : ℝ) : ℝ → ℝ := fun x ↦ m * x + c

theorem linear_function_properties (f : ℝ → ℝ) (m c : ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, f x + 2 = 3 * k * x)
  (h2 : f 1 = 4)
  (h3 : f = LinearFunction m c) :
  (f = LinearFunction 6 (-2)) ∧ 
  (∀ a b : ℝ, f (-1) = a ∧ f 2 = b → a < b) := by
  sorry

end linear_function_properties_l2230_223089


namespace product_of_one_plus_greater_than_eight_l2230_223090

theorem product_of_one_plus_greater_than_eight
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_prod : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 8 := by
  sorry

end product_of_one_plus_greater_than_eight_l2230_223090


namespace seashells_given_away_l2230_223000

/-- Represents the number of seashells Maura collected and gave away -/
structure SeashellCollection where
  total : ℕ
  left : ℕ
  given : ℕ

/-- Theorem stating that the number of seashells given away is the difference between total and left -/
theorem seashells_given_away (collection : SeashellCollection) 
  (h1 : collection.total = 75)
  (h2 : collection.left = 57)
  (h3 : collection.given = collection.total - collection.left) :
  collection.given = 18 := by
  sorry

end seashells_given_away_l2230_223000


namespace staples_remaining_after_stapling_l2230_223063

/-- Calculates the number of staples left in a stapler after stapling reports. -/
def staples_left (initial_staples : ℕ) (reports_stapled : ℕ) : ℕ :=
  initial_staples - reports_stapled

/-- Converts dozens to individual units. -/
def dozens_to_units (dozens : ℕ) : ℕ :=
  dozens * 12

theorem staples_remaining_after_stapling :
  let initial_staples := 50
  let reports_in_dozens := 3
  let reports_stapled := dozens_to_units reports_in_dozens
  staples_left initial_staples reports_stapled = 14 := by
sorry

end staples_remaining_after_stapling_l2230_223063


namespace jucas_marbles_l2230_223055

theorem jucas_marbles :
  ∃! B : ℕ, 0 < B ∧ B < 800 ∧
  B % 3 = 2 ∧
  B % 4 = 3 ∧
  B % 5 = 4 ∧
  B % 7 = 6 ∧
  B % 20 = 19 ∧
  B = 419 :=
by sorry

end jucas_marbles_l2230_223055


namespace product_less_than_sum_plus_one_l2230_223006

theorem product_less_than_sum_plus_one (a₁ a₂ : ℝ) 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ < a₁ + a₂ + 1 := by
  sorry

#check product_less_than_sum_plus_one

end product_less_than_sum_plus_one_l2230_223006


namespace decimal_to_fraction_l2230_223058

theorem decimal_to_fraction : 
  (3.56 : ℚ) = 89 / 25 := by sorry

end decimal_to_fraction_l2230_223058


namespace equation_solutions_l2230_223099

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ∧ 
           (x^2 + 13*x - 16 ≠ 0) ∧ (x^2 + 4*x - 16 ≠ 0) ∧ (x^2 - 15*x - 16 ≠ 0)} = 
  {1, -16, 4, -4} := by
sorry

end equation_solutions_l2230_223099


namespace fraction_problem_l2230_223027

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N * f^2 = 6^3 ∧ N * f^2 = 7776 → f = 1/6 := by
sorry

end fraction_problem_l2230_223027


namespace lisa_spoon_count_l2230_223091

theorem lisa_spoon_count :
  let num_children : ℕ := 6
  let baby_spoons_per_child : ℕ := 4
  let decorative_spoons : ℕ := 4
  let large_spoons : ℕ := 20
  let dessert_spoons : ℕ := 10
  let teaspoons : ℕ := 25
  
  let total_baby_spoons := num_children * baby_spoons_per_child
  let total_special_spoons := total_baby_spoons + decorative_spoons
  let total_new_spoons := large_spoons + dessert_spoons + teaspoons
  let total_spoons := total_special_spoons + total_new_spoons

  total_spoons = 83 := by
sorry

end lisa_spoon_count_l2230_223091


namespace perpendicular_vectors_l2230_223007

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, -3)

theorem perpendicular_vectors (k : ℝ) : 
  (k • a - 2 • b) • a = 0 → k = -1 := by
  sorry

end perpendicular_vectors_l2230_223007


namespace leftover_sets_problem_l2230_223025

/-- Given a total number of crayons, number of friends, and crayons per set,
    calculate the number of complete sets left over after distributing one set to each friend. -/
def leftover_sets (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_set : ℕ) : ℕ :=
  (total_crayons / crayons_per_set) % num_friends

theorem leftover_sets_problem :
  leftover_sets 210 30 5 = 12 := by
  sorry

#eval leftover_sets 210 30 5

end leftover_sets_problem_l2230_223025


namespace cubic_sum_theorem_l2230_223056

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end cubic_sum_theorem_l2230_223056


namespace bird_count_after_changes_l2230_223013

/-- Represents the number of birds of each type on the fence -/
structure BirdCount where
  sparrows : ℕ
  storks : ℕ
  pigeons : ℕ
  swallows : ℕ

/-- Calculates the total number of birds -/
def totalBirds (birds : BirdCount) : ℕ :=
  birds.sparrows + birds.storks + birds.pigeons + birds.swallows

/-- Represents the changes in bird population -/
structure BirdChanges where
  sparrowsJoined : ℕ
  swallowsJoined : ℕ
  pigeonsLeft : ℕ

/-- Applies changes to the bird population -/
def applyChanges (initial : BirdCount) (changes : BirdChanges) : BirdCount :=
  { sparrows := initial.sparrows + changes.sparrowsJoined,
    storks := initial.storks,
    pigeons := initial.pigeons - changes.pigeonsLeft,
    swallows := initial.swallows + changes.swallowsJoined }

theorem bird_count_after_changes 
  (initial : BirdCount)
  (changes : BirdChanges)
  (h_initial : initial = { sparrows := 3, storks := 2, pigeons := 4, swallows := 0 })
  (h_changes : changes = { sparrowsJoined := 3, swallowsJoined := 5, pigeonsLeft := 2 }) :
  totalBirds (applyChanges initial changes) = 15 := by
  sorry


end bird_count_after_changes_l2230_223013


namespace simplify_fraction_expression_l2230_223016

theorem simplify_fraction_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -m - n := by
  sorry

end simplify_fraction_expression_l2230_223016


namespace fraction_simplification_l2230_223086

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 2/7) 
  (hy : y = 8/11) : 
  (7*x + 11*y) / (77*x*y) = 5/8 := by
  sorry

end fraction_simplification_l2230_223086


namespace min_sum_dimensions_l2230_223030

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 3003 → l + w + h ≥ 45 := by
  sorry

end min_sum_dimensions_l2230_223030


namespace inequality_solution_l2230_223033

theorem inequality_solution (x : ℝ) : 
  (x - 1)^2 < 12 - x ↔ (1 - 3 * Real.sqrt 5) / 2 < x ∧ x < (1 + 3 * Real.sqrt 5) / 2 := by
  sorry

end inequality_solution_l2230_223033


namespace rhombus_longest_diagonal_l2230_223001

theorem rhombus_longest_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 108 →
  ratio_long = 3 →
  ratio_short = 2 →
  let diagonal_long := ratio_long * (2 * area / (ratio_long * ratio_short)) ^ (1/2 : ℝ)
  diagonal_long = 18 := by
  sorry

end rhombus_longest_diagonal_l2230_223001


namespace quadratic_inequality_range_l2230_223044

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end quadratic_inequality_range_l2230_223044


namespace largest_prime_divisor_of_101110111_base_5_l2230_223065

def base_five_to_decimal (n : ℕ) : ℕ := 
  5^8 + 5^6 + 5^5 + 5^4 + 5^3 + 5^2 + 5^1 + 5^0

theorem largest_prime_divisor_of_101110111_base_5 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ base_five_to_decimal 101110111 ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ base_five_to_decimal 101110111 → q ≤ p :=
by sorry

end largest_prime_divisor_of_101110111_base_5_l2230_223065
