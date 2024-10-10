import Mathlib

namespace min_rooms_for_departments_l529_52914

def minRooms (d1 d2 d3 : Nat) : Nat :=
  let gcd := Nat.gcd (Nat.gcd d1 d2) d3
  (d1 / gcd) + (d2 / gcd) + (d3 / gcd)

theorem min_rooms_for_departments :
  minRooms 72 58 24 = 77 := by
  sorry

end min_rooms_for_departments_l529_52914


namespace odd_function_iff_a_b_zero_l529_52935

def f (x a b : ℝ) : ℝ := x * abs (x - a) + b

theorem odd_function_iff_a_b_zero (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end odd_function_iff_a_b_zero_l529_52935


namespace circle_divides_rectangle_sides_l529_52954

/-- A circle touching two adjacent sides of a rectangle --/
structure CircleTouchingRectangle where
  radius : ℝ
  rect_side1 : ℝ
  rect_side2 : ℝ
  (radius_positive : 0 < radius)
  (rect_sides_positive : 0 < rect_side1 ∧ 0 < rect_side2)
  (radius_fits : radius < rect_side1 ∧ radius < rect_side2)

/-- The segments into which the circle divides the rectangle sides --/
structure RectangleSegments where
  seg1 : ℝ
  seg2 : ℝ
  seg3 : ℝ
  seg4 : ℝ
  seg5 : ℝ
  seg6 : ℝ

/-- Theorem stating how the circle divides the rectangle sides --/
theorem circle_divides_rectangle_sides (c : CircleTouchingRectangle) 
  (h : c.radius = 26 ∧ c.rect_side1 = 36 ∧ c.rect_side2 = 60) :
  ∃ (s : RectangleSegments), 
    s.seg1 = 26 ∧ s.seg2 = 34 ∧ 
    s.seg3 = 26 ∧ s.seg4 = 10 ∧ 
    s.seg5 = 2 ∧ s.seg6 = 48 :=
sorry

end circle_divides_rectangle_sides_l529_52954


namespace election_winner_margin_l529_52968

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) : 
  winner_percentage = 62 / 100 ∧ 
  winner_votes = 775 ∧ 
  winner_votes = (winner_percentage * total_votes).floor →
  winner_votes - (total_votes - winner_votes) = 300 := by
sorry

end election_winner_margin_l529_52968


namespace probability_king_or_queen_l529_52973

-- Define the structure of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

-- Define the properties of a standard deck
def is_standard_deck (d : StandardDeck) : Prop :=
  d.total_cards = 52 ∧
  d.num_ranks = 13 ∧
  d.num_suits = 4 ∧
  d.num_kings = 4 ∧
  d.num_queens = 4

-- Theorem statement
theorem probability_king_or_queen (d : StandardDeck) 
  (h : is_standard_deck d) : 
  (d.num_kings + d.num_queens : ℚ) / d.total_cards = 2 / 13 := by
  sorry

end probability_king_or_queen_l529_52973


namespace triangle_trig_max_l529_52979

open Real

theorem triangle_trig_max (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  π/4 < B ∧ B < π/2 ∧
  a * cos B - b * cos A = (3/5) * c →
  ∃ (max_val : ℝ), max_val = -512 ∧ 
    ∀ x, x = tan (2*B) * (tan A)^3 → x ≤ max_val :=
sorry

end triangle_trig_max_l529_52979


namespace bake_sale_cookies_l529_52948

/-- The number of cookies in each box -/
def cookies_per_box : ℕ := 48

/-- The number of boxes Abigail collected -/
def abigail_boxes : ℕ := 2

/-- The number of quarter boxes Grayson collected -/
def grayson_quarter_boxes : ℕ := 3

/-- The number of boxes Olivia collected -/
def olivia_boxes : ℕ := 3

/-- The total number of cookies collected -/
def total_cookies : ℕ := 276

theorem bake_sale_cookies : 
  cookies_per_box * abigail_boxes + 
  (cookies_per_box / 4) * grayson_quarter_boxes + 
  cookies_per_box * olivia_boxes = total_cookies := by
sorry

end bake_sale_cookies_l529_52948


namespace intersection_of_A_and_B_l529_52962

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| > 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l529_52962


namespace rotation_maps_points_l529_52952

-- Define points in R²
def C : ℝ × ℝ := (3, -2)
def C' : ℝ × ℝ := (-3, 2)
def D : ℝ × ℝ := (4, -5)
def D' : ℝ × ℝ := (-4, 5)

-- Define rotation by 180°
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation_maps_points :
  rotate180 C = C' ∧ rotate180 D = D' :=
sorry

end rotation_maps_points_l529_52952


namespace total_fruits_l529_52970

theorem total_fruits (apples bananas grapes : ℕ) 
  (h1 : apples = 5) 
  (h2 : bananas = 4) 
  (h3 : grapes = 6) : 
  apples + bananas + grapes = 15 := by
  sorry

end total_fruits_l529_52970


namespace whole_number_between_fractions_l529_52996

theorem whole_number_between_fractions (M : ℤ) : 
  (5 < (M : ℚ) / 4) ∧ ((M : ℚ) / 4 < 5.5) → M = 21 := by
  sorry

end whole_number_between_fractions_l529_52996


namespace parallelogram_properties_l529_52901

-- Define the parallelogram vertices as complex numbers
def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

-- Define the parallelogram
def parallelogram (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

-- Theorem statement
theorem parallelogram_properties (h : parallelogram A B C) :
  ∃ D : ℂ,
    D = 4 + 3 * Complex.I ∧
    Complex.abs (C - A) = Real.sqrt 17 ∧
    Complex.abs (D - B) = Real.sqrt 18 :=
sorry

end parallelogram_properties_l529_52901


namespace complement_intersection_theorem_l529_52944

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set N
def N : Finset Nat := {2, 4, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {3} := by sorry

end complement_intersection_theorem_l529_52944


namespace regular_polygon_interior_angle_sum_l529_52967

theorem regular_polygon_interior_angle_sum :
  ∀ n : ℕ,
  n > 2 →
  (360 : ℝ) / n = 20 →
  (n - 2) * 180 = 2880 :=
by
  sorry

end regular_polygon_interior_angle_sum_l529_52967


namespace apples_total_weight_l529_52917

def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def plum_weight : ℕ := 2
def bag_capacity : ℕ := 49
def num_bags : ℕ := 5

def fruit_set_weight : ℕ := apple_weight + orange_weight + plum_weight

def fruits_per_bag : ℕ := (bag_capacity / fruit_set_weight) * fruit_set_weight

theorem apples_total_weight :
  fruits_per_bag / fruit_set_weight * apple_weight * num_bags = 80 := by sorry

end apples_total_weight_l529_52917


namespace simplify_fraction_l529_52939

theorem simplify_fraction : (140 : ℚ) / 9800 * 35 = 1 / 70 := by sorry

end simplify_fraction_l529_52939


namespace b_join_time_l529_52984

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment in Rupees -/
def aInvestment : ℕ := 45000

/-- Represents B's initial investment in Rupees -/
def bInvestment : ℕ := 27000

/-- Represents the ratio of profit sharing between A and B -/
def profitRatio : ℚ := 2 / 1

/-- 
Proves that B joined 2 months after A started the business, given the initial investments
and profit ratio.
-/
theorem b_join_time : 
  ∀ x : ℕ, 
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = profitRatio → 
  x = 2 :=
by sorry

end b_join_time_l529_52984


namespace max_k_value_l529_52978

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * (x^2/y^2 + 2 + y^2/x^2) + k^3 * (x/y + y/x)) :
  k ≤ 4 * Real.sqrt 2 - 4 :=
by sorry

end max_k_value_l529_52978


namespace pet_store_spiders_l529_52963

theorem pet_store_spiders (initial_birds initial_puppies initial_cats : ℕ)
  (initial_spiders : ℕ) (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_cats = 5 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  total_left = (initial_birds - sold_birds) + (initial_puppies - adopted_puppies) +
               initial_cats + (initial_spiders - loose_spiders) →
  initial_spiders = 15 :=
by sorry

end pet_store_spiders_l529_52963


namespace power_difference_equality_l529_52971

theorem power_difference_equality : (3^4)^4 - (4^3)^3 = 42792577 := by
  sorry

end power_difference_equality_l529_52971


namespace binary_calculation_l529_52918

theorem binary_calculation : 
  (0b110101 * 0b1101) + 0b1010 = 0b10010111111 := by sorry

end binary_calculation_l529_52918


namespace number_less_than_hundred_million_l529_52900

theorem number_less_than_hundred_million :
  ∃ x : ℕ,
    x < 100000000 ∧
    x + 1000000 = 100000000 ∧
    x = 99000000 ∧
    x / 1000000 = 99 := by
  sorry

end number_less_than_hundred_million_l529_52900


namespace parabola_vertex_l529_52957

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x + 1)^2 - 6

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, -6)

/-- Theorem: The vertex of the parabola y = -2(x+1)^2 - 6 is at the point (-1, -6) -/
theorem parabola_vertex :
  let (h, k) := vertex
  ∀ x y, parabola x y → (x - h)^2 ≤ (y - k) / (-2) := by
  sorry

end parabola_vertex_l529_52957


namespace yellow_bows_count_l529_52926

theorem yellow_bows_count (total : ℚ) :
  (1 / 6 : ℚ) * total +  -- yellow bows
  (1 / 3 : ℚ) * total +  -- purple bows
  (1 / 8 : ℚ) * total +  -- orange bows
  40 = total →           -- black bows
  (1 / 6 : ℚ) * total = 160 / 9 := by
sorry

end yellow_bows_count_l529_52926


namespace absolute_value_equation_solutions_l529_52923

theorem absolute_value_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | |x + 1| = |x - 2| + |x - 5| + |x - 6|} ∧ s = {4, 7} := by
  sorry

end absolute_value_equation_solutions_l529_52923


namespace four_common_tangents_l529_52920

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y - 2 = 0

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

/-- The number of common tangent lines between C₁ and C₂ -/
def num_common_tangents : ℕ := 4

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 4 -/
theorem four_common_tangents : num_common_tangents = 4 := by
  sorry

end four_common_tangents_l529_52920


namespace tangent_double_angle_identity_l529_52989

theorem tangent_double_angle_identity (α : Real) (h : 0 < α ∧ α < π/4) : 
  Real.tan (2 * α) / Real.tan α = 1 + 1 / Real.cos (2 * α) := by
  sorry

end tangent_double_angle_identity_l529_52989


namespace no_solution_iff_a_geq_bound_l529_52910

theorem no_solution_iff_a_geq_bound (a : ℝ) :
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) ↔ a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end no_solution_iff_a_geq_bound_l529_52910


namespace steps_to_distance_l529_52921

/-- Given that 625 steps correspond to 500 meters, prove that 10,000 steps at the same rate will result in a distance of 8 km. -/
theorem steps_to_distance (steps_short : ℕ) (distance_short : ℝ) (steps_long : ℕ) :
  steps_short = 625 →
  distance_short = 500 →
  steps_long = 10000 →
  (distance_short / steps_short) * steps_long = 8000 :=
by sorry

end steps_to_distance_l529_52921


namespace second_year_percentage_correct_l529_52916

/-- The number of second-year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second-year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 673

/-- The percentage of second-year students in the faculty -/
def second_year_percentage : ℚ :=
  (numeric_methods + automatic_control - both_subjects : ℚ) / total_students * 100

theorem second_year_percentage_correct :
  second_year_percentage = (250 + 423 - 134 : ℚ) / 673 * 100 :=
by sorry

end second_year_percentage_correct_l529_52916


namespace arrangement_theorems_l529_52949

/-- The number of men in the group -/
def num_men : ℕ := 6

/-- The number of women in the group -/
def num_women : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- Calculate the number of arrangements with no two women next to each other -/
def arrangements_no_adjacent_women : ℕ := sorry

/-- Calculate the number of arrangements with Man A not first and Man B not last -/
def arrangements_a_not_first_b_not_last : ℕ := sorry

/-- Calculate the number of arrangements with fixed order of Men A, B, and C -/
def arrangements_fixed_abc : ℕ := sorry

/-- Calculate the number of arrangements with Man A to the left of Man B -/
def arrangements_a_left_of_b : ℕ := sorry

theorem arrangement_theorems :
  (arrangements_no_adjacent_women = num_men.factorial * (num_women.choose (num_men + 1))) ∧
  (arrangements_a_not_first_b_not_last = total_people.factorial - 2 * (total_people - 1).factorial + (total_people - 2).factorial) ∧
  (arrangements_fixed_abc = total_people.factorial / 6) ∧
  (arrangements_a_left_of_b = total_people.factorial / 2) := by sorry

end arrangement_theorems_l529_52949


namespace sum_of_reciprocals_l529_52958

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 36) (h3 : x > 0) (h4 : y > 0) :
  1 / x + 1 / y = 1 / 3 := by
sorry

end sum_of_reciprocals_l529_52958


namespace fraction_simplification_l529_52924

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (2 / y + 1 / x) / (1 / x) = 13 / 5 := by
  sorry

end fraction_simplification_l529_52924


namespace maxwell_walking_speed_l529_52997

-- Define the given conditions
def total_distance : ℝ := 65
def maxwell_distance : ℝ := 26
def brad_speed : ℝ := 3

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem to prove
theorem maxwell_walking_speed :
  (maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) →
  maxwell_speed = 2 := by
  sorry

end maxwell_walking_speed_l529_52997


namespace equal_roots_quadratic_l529_52946

/-- 
Given a quadratic equation x^2 - 6x + k = 0 with two equal real roots,
prove that k = 9
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 6*y + k = 0 → y = x) → 
  k = 9 := by
  sorry

end equal_roots_quadratic_l529_52946


namespace largest_number_proof_l529_52993

theorem largest_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (Nat.gcd a b = 42) → 
  (∃ k : ℕ, Nat.lcm a b = 42 * 11 * 12 * k) →
  (max a b = 504) := by
sorry

end largest_number_proof_l529_52993


namespace sector_perimeter_to_circumference_ratio_l529_52988

theorem sector_perimeter_to_circumference_ratio (r : ℝ) (hr : r > 0) :
  let circumference := 2 * π * r
  let sector_arc_length := circumference / 3
  let sector_perimeter := sector_arc_length + 2 * r
  sector_perimeter / circumference = (π + 3) / (3 * π) := by
sorry

end sector_perimeter_to_circumference_ratio_l529_52988


namespace quadratic_one_root_l529_52998

theorem quadratic_one_root (m : ℝ) : m > 0 ∧ 
  (∃! x : ℝ, x^2 + 6*m*x + 3*m = 0) ↔ m = 1/3 := by
  sorry

end quadratic_one_root_l529_52998


namespace car_distance_calculation_car_distance_is_432_l529_52911

/-- Given a car's journey with known time and alternative speed, calculate the distance. -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  let new_time := initial_time * time_ratio
  let distance := new_speed * new_time
  distance

/-- Prove that the distance covered by the car is 432 km. -/
theorem car_distance_is_432 :
  car_distance_calculation 6 48 (3/2) = 432 := by
  sorry

end car_distance_calculation_car_distance_is_432_l529_52911


namespace mike_picked_52_peaches_l529_52947

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mike picked 52 peaches -/
theorem mike_picked_52_peaches (initial final : ℕ) 
  (h1 : initial = 34) 
  (h2 : final = 86) : 
  peaches_picked initial final = 52 := by
  sorry

end mike_picked_52_peaches_l529_52947


namespace point_translation_l529_52930

def initial_point : ℝ × ℝ := (-5, 1)
def x_translation : ℝ := 2
def y_translation : ℝ := -4

theorem point_translation (P : ℝ × ℝ) (dx dy : ℝ) :
  P = initial_point →
  (P.1 + dx, P.2 + dy) = (-3, -3) :=
by sorry

end point_translation_l529_52930


namespace range_of_t_when_p_range_of_t_when_p_xor_q_l529_52976

-- Define the propositions
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 
  t ≠ 4 ∧ t ≠ 2 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y, x^2 / (4 - t) + y^2 / (t - 2) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem 1
theorem range_of_t_when_p (t : ℝ) : p t → t > 5/2 := by sorry

-- Theorem 2
theorem range_of_t_when_p_xor_q (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by sorry

end range_of_t_when_p_range_of_t_when_p_xor_q_l529_52976


namespace min_red_to_blue_l529_52929

/-- Represents the color of a chameleon -/
inductive Color
| Red
| Blue
| Other1
| Other2
| Other3

/-- Represents the result of a bite interaction between two chameleons -/
def bite_result : Color → Color → Color := sorry

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- The number of colors available -/
def num_colors : Nat := 5

/-- The given number of red chameleons that can become blue -/
def given_red_count : Nat := 2023

/-- Checks if a sequence of bites transforms all chameleons to blue -/
def all_blue (initial : List Color) (sequence : BiteSequence) : Prop := sorry

/-- The theorem to be proved -/
theorem min_red_to_blue :
  ∀ (k : Nat),
  (k ≥ 5) →
  (∃ (sequence : BiteSequence), all_blue (List.replicate k Color.Red) sequence) ∧
  (∀ (j : Nat), j < 5 →
    ¬∃ (sequence : BiteSequence), all_blue (List.replicate j Color.Red) sequence) :=
sorry

end min_red_to_blue_l529_52929


namespace remaining_money_l529_52992

def initial_amount : ℕ := 11
def spent_amount : ℕ := 2
def lost_amount : ℕ := 6

theorem remaining_money :
  initial_amount - spent_amount - lost_amount = 3 := by sorry

end remaining_money_l529_52992


namespace sum_of_imaginary_parts_zero_l529_52960

theorem sum_of_imaginary_parts_zero (z : ℂ) : 
  (z^2 - 2*z = -1 + Complex.I) → 
  (∃ z₁ z₂ : ℂ, (z₁^2 - 2*z₁ = -1 + Complex.I) ∧ 
                (z₂^2 - 2*z₂ = -1 + Complex.I) ∧ 
                (z₁ ≠ z₂) ∧
                (Complex.im z₁ + Complex.im z₂ = 0)) :=
by sorry

end sum_of_imaginary_parts_zero_l529_52960


namespace f_decreasing_on_interval_l529_52987

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, 
    ∀ y ∈ Set.Ioo (-1 : ℝ) 3, 
      x < y → f x > f y :=
by sorry

end f_decreasing_on_interval_l529_52987


namespace line_parameterization_l529_52964

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 10 = 2 * f t - 30) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
sorry

end line_parameterization_l529_52964


namespace intersection_of_A_and_B_l529_52966

def A : Set ℝ := {1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | ∃ y ∈ A, y = 2 * x}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l529_52966


namespace easter_egg_hunt_problem_l529_52908

/-- Represents the number of eggs of each size found by a child -/
structure EggCount where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total points for a given EggCount -/
def totalPoints (eggs : EggCount) : ℕ :=
  eggs.small + 3 * eggs.medium + 5 * eggs.large

/-- The Easter egg hunt problem -/
theorem easter_egg_hunt_problem :
  let kevin := EggCount.mk 5 0 3
  let bonnie := EggCount.mk 13 7 2
  let george := EggCount.mk 9 6 1
  let cheryl := EggCount.mk 56 30 15
  totalPoints cheryl - (totalPoints kevin + totalPoints bonnie + totalPoints george) = 125 := by
  sorry


end easter_egg_hunt_problem_l529_52908


namespace genetically_modified_microorganisms_percentage_l529_52959

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysicsDegrees : ℝ

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem genetically_modified_microorganisms_percentage 
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 13)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.industrialLubricants = 8)
  (h5 : budget.basicAstrophysicsDegrees = 39.6) :
  100 - (budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
         budget.industrialLubricants + (budget.basicAstrophysicsDegrees / 360 * 100)) = 29 := by
  sorry

end genetically_modified_microorganisms_percentage_l529_52959


namespace circle_satisfies_conditions_l529_52907

/-- The line on which the center of the circle lies -/
def center_line (x y : ℝ) : Prop := x - y - 4 = 0

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

/-- The equation of the circle we want to prove -/
def target_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 7*y - 32 = 0

/-- Theorem stating that the target_circle satisfies the given conditions -/
theorem circle_satisfies_conditions :
  ∃ (h k : ℝ), 
    (center_line h k) ∧ 
    (∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → target_circle x y) ∧
    ((h - 1/2)^2 + (k - 7/2)^2 = (33/2)^2) :=
sorry

end circle_satisfies_conditions_l529_52907


namespace normal_distribution_std_dev_l529_52981

theorem normal_distribution_std_dev (μ σ x : ℝ) (hμ : μ = 17.5) (hσ : σ = 2.5) (hx : x = 12.5) :
  (x - μ) / σ = -2 := by
  sorry

end normal_distribution_std_dev_l529_52981


namespace math_majors_consecutive_probability_l529_52975

/-- The number of people sitting around the table -/
def total_people : ℕ := 11

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 3

/-- The probability of math majors sitting consecutively -/
def consecutive_math_prob : ℚ := 1 / 42

theorem math_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := Nat.factorial (total_people - math_majors) * Nat.factorial math_majors
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_math_prob := by
  sorry

#check math_majors_consecutive_probability

end math_majors_consecutive_probability_l529_52975


namespace divisibility_by_eleven_l529_52950

def seven_digit_number (n : ℕ) : ℕ := 7010000 + n * 1000 + 864

theorem divisibility_by_eleven (n : ℕ) :
  (seven_digit_number n) % 11 = 0 ↔ n = 3 := by
  sorry

end divisibility_by_eleven_l529_52950


namespace continuity_at_seven_l529_52991

/-- The function f(x) = 4x^2 + 6 is continuous at x₀ = 7 -/
theorem continuity_at_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - 7| < δ → |4 * x^2 + 6 - (4 * 7^2 + 6)| < ε := by
  sorry

end continuity_at_seven_l529_52991


namespace pass_rate_two_procedures_l529_52915

theorem pass_rate_two_procedures (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  let pass_rate := (1 - a) * (1 - b)
  0 ≤ pass_rate ∧ pass_rate ≤ 1 :=
by sorry

end pass_rate_two_procedures_l529_52915


namespace triangle_with_angle_ratio_not_necessarily_right_l529_52912

/-- Triangle ABC with angles in the ratio 3:4:5 is not necessarily a right triangle -/
theorem triangle_with_angle_ratio_not_necessarily_right :
  ∀ (A B C : ℝ),
  (A + B + C = 180) →
  (A : ℝ) / 3 = (B : ℝ) / 4 →
  (B : ℝ) / 4 = (C : ℝ) / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end triangle_with_angle_ratio_not_necessarily_right_l529_52912


namespace square_perimeter_7m_l529_52909

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter_7m : 
  ∀ (s : ℝ), s = 7 → 4 * s = 28 := by
  sorry

end square_perimeter_7m_l529_52909


namespace relationship_abc_l529_52961

theorem relationship_abc (a b c : ℝ) 
  (h : Real.exp a + a = Real.log b + b ∧ Real.log b + b = Real.sqrt c + c ∧ Real.sqrt c + c = Real.sin 1) : 
  a < c ∧ c < b := by
  sorry

end relationship_abc_l529_52961


namespace intersection_of_A_and_B_l529_52983

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end intersection_of_A_and_B_l529_52983


namespace total_tickets_sold_l529_52982

theorem total_tickets_sold (adult_tickets student_tickets : ℕ) 
  (h1 : adult_tickets = 410)
  (h2 : student_tickets = 436) :
  adult_tickets + student_tickets = 846 := by
  sorry

#check total_tickets_sold

end total_tickets_sold_l529_52982


namespace total_painting_cost_l529_52990

/-- Calculate the last term of an arithmetic sequence -/
def lastTerm (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- Count the number of digits in a natural number -/
def digitCount (n : ℕ) : ℕ :=
  if n < 10 then 1 else if n < 100 then 2 else 3

/-- Calculate the cost of painting numbers for one side of the street -/
def sideCost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  let lastNum := lastTerm a₁ d n
  let twoDigitCount := (min 99 lastNum - a₁) / d + 1
  let threeDigitCount := n - twoDigitCount
  2 * (2 * twoDigitCount + 3 * threeDigitCount)

/-- The main theorem stating the total cost for painting all house numbers -/
theorem total_painting_cost : 
  sideCost 5 7 30 + sideCost 6 8 30 = 312 := by sorry

end total_painting_cost_l529_52990


namespace negation_equivalence_l529_52919

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ > 1) ↔
  (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end negation_equivalence_l529_52919


namespace E_is_true_l529_52942

-- Define the statements as propositions
variable (A B C D E : Prop)

-- Define the condition that only one statement is true
def only_one_true (A B C D E : Prop) : Prop :=
  (A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ E)

-- Define the content of each statement
def statement_definitions (A B C D E : Prop) : Prop :=
  (A ↔ B) ∧
  (B ↔ ¬E) ∧
  (C ↔ (A ∧ B ∧ C ∧ D ∧ E)) ∧
  (D ↔ (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E)) ∧
  (E ↔ ¬A)

-- Theorem stating that E is the only true statement
theorem E_is_true (A B C D E : Prop) 
  (h1 : only_one_true A B C D E) 
  (h2 : statement_definitions A B C D E) : 
  E ∧ ¬A ∧ ¬B ∧ ¬C ∧ ¬D :=
sorry

end E_is_true_l529_52942


namespace exactly_one_greater_than_one_l529_52938

theorem exactly_one_greater_than_one (x₁ x₂ x₃ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
  (h_product : x₁ * x₂ * x₃ = 1)
  (h_sum : x₁ + x₂ + x₃ > 1/x₁ + 1/x₂ + 1/x₃) :
  (x₁ > 1 ∧ x₂ ≤ 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ > 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ ≤ 1 ∧ x₃ > 1) :=
by sorry

end exactly_one_greater_than_one_l529_52938


namespace negative_sqrt_13_less_than_negative_3_l529_52902

theorem negative_sqrt_13_less_than_negative_3 : -Real.sqrt 13 < -3 := by
  sorry

end negative_sqrt_13_less_than_negative_3_l529_52902


namespace expression_evaluation_l529_52936

theorem expression_evaluation (x : ℝ) (h : x < 2) :
  Real.sqrt ((x - 2) / (1 - (x - 3) / (x - 2))) = (2 - x) / Real.sqrt 3 := by
  sorry

end expression_evaluation_l529_52936


namespace number_puzzle_l529_52913

theorem number_puzzle : ∃ N : ℚ, N = (3/8) * N + (1/4) * N + 15 ∧ N = 40 := by
  sorry

end number_puzzle_l529_52913


namespace ant_movement_l529_52969

-- Define the type for a 2D position
def Position := ℝ × ℝ

-- Define the initial position
def initial_position : Position := (-2, 4)

-- Define the horizontal movement
def horizontal_movement : ℝ := 3

-- Define the vertical movement
def vertical_movement : ℝ := -2

-- Define the function to calculate the final position
def final_position (initial : Position) (horizontal : ℝ) (vertical : ℝ) : Position :=
  (initial.1 + horizontal, initial.2 + vertical)

-- Theorem statement
theorem ant_movement :
  final_position initial_position horizontal_movement vertical_movement = (1, 2) := by
  sorry

end ant_movement_l529_52969


namespace beach_probability_l529_52977

theorem beach_probability (total : ℕ) (sunglasses : ℕ) (caps : ℕ) (cap_and_sunglasses_prob : ℚ) :
  total = 100 →
  sunglasses = 70 →
  caps = 60 →
  cap_and_sunglasses_prob = 2/3 →
  (cap_and_sunglasses_prob * caps : ℚ) / sunglasses = 4/7 := by
  sorry

end beach_probability_l529_52977


namespace palindromic_four_digit_squares_l529_52925

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def ends_with_0_4_or_6 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 6

def satisfies_conditions (n : ℕ) : Prop :=
  is_square n ∧ is_four_digit n ∧ is_palindrome n ∧ ends_with_0_4_or_6 n

theorem palindromic_four_digit_squares :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n :=
sorry

end palindromic_four_digit_squares_l529_52925


namespace smaller_solution_quadratic_l529_52986

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 15*x + 36 = 0 ∧ x ≠ -3 → x = -12 :=
by sorry

end smaller_solution_quadratic_l529_52986


namespace eight_faucets_fill_time_correct_l529_52994

/-- The time (in seconds) it takes for eight faucets to fill a 50-gallon tank,
    given that four faucets fill a 200-gallon tank in 8 minutes and all faucets
    dispense water at the same rate. -/
def eight_faucets_fill_time : ℕ := by sorry

/-- Four faucets fill a 200-gallon tank in 8 minutes. -/
def four_faucets_fill_time : ℕ := 8 * 60  -- in seconds

/-- The volume of the tank filled by four faucets. -/
def four_faucets_volume : ℕ := 200  -- in gallons

/-- The volume of the tank to be filled by eight faucets. -/
def eight_faucets_volume : ℕ := 50  -- in gallons

/-- All faucets dispense water at the same rate. -/
axiom faucets_equal_rate : True

theorem eight_faucets_fill_time_correct :
  eight_faucets_fill_time = 60 := by sorry

end eight_faucets_fill_time_correct_l529_52994


namespace smallest_five_digit_multiple_with_16_divisors_l529_52922

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def last_three_digits (n : ℕ) : ℕ := n % 1000

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_five_digit_multiple_with_16_divisors :
  ∃ (n : ℕ), is_five_digit n ∧ 2014 ∣ n ∧ count_divisors (last_three_digits n) = 16 ∧
  ∀ (m : ℕ), is_five_digit m ∧ 2014 ∣ m ∧ count_divisors (last_three_digits m) = 16 → n ≤ m :=
by sorry

end smallest_five_digit_multiple_with_16_divisors_l529_52922


namespace magician_balls_l529_52906

/-- Represents the number of balls in the box after each operation -/
def BallCount : ℕ → ℕ
  | 0 => 7  -- Initial count
  | n + 1 => BallCount n + 6 * (BallCount n - 1)  -- After each operation

/-- The form that the ball count must follow -/
def ValidForm (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k + 7

theorem magician_balls :
  (∀ n : ℕ, ValidForm (BallCount n)) ∧
  ValidForm 1993 ∧
  ¬ValidForm 1990 ∧
  ¬ValidForm 1991 ∧
  ¬ValidForm 1992 := by sorry

end magician_balls_l529_52906


namespace phi_subset_singleton_zero_l529_52933

-- Define Φ as a set
variable (Φ : Set ℕ)

-- Theorem stating that Φ is a subset of {0}
theorem phi_subset_singleton_zero : Φ ⊆ {0} := by
  sorry

end phi_subset_singleton_zero_l529_52933


namespace estimate_percentage_negative_attitude_l529_52932

theorem estimate_percentage_negative_attitude 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (negative_attitude_count : ℕ) 
  (h1 : total_population = 2500)
  (h2 : sample_size = 400)
  (h3 : negative_attitude_count = 360) :
  (negative_attitude_count : ℝ) / (sample_size : ℝ) * 100 = 90 := by
sorry

end estimate_percentage_negative_attitude_l529_52932


namespace two_cubic_feet_to_cubic_inches_l529_52995

/-- Converts cubic feet to cubic inches -/
def cubic_feet_to_cubic_inches (cf : ℝ) : ℝ := cf * (12^3)

/-- Theorem stating that 2 cubic feet equals 3456 cubic inches -/
theorem two_cubic_feet_to_cubic_inches : 
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end two_cubic_feet_to_cubic_inches_l529_52995


namespace complex_product_theorem_l529_52937

theorem complex_product_theorem (a b : ℝ) :
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk a (-b)
  let z₃ : ℂ := Complex.mk (-a) b
  let z₄ : ℂ := Complex.mk (-a) (-b)
  (z₁ * z₂ * z₃ * z₄).re = (a^2 + b^2)^2 ∧ (z₁ * z₂ * z₃ * z₄).im = 0 :=
by sorry

end complex_product_theorem_l529_52937


namespace smallest_n_for_inequality_l529_52934

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x + y + z)^2 ≤ n * (x^2 + y^2 + z^2)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x + y + z)^2 > m * (x^2 + y^2 + z^2)) :=
by sorry

end smallest_n_for_inequality_l529_52934


namespace greatest_fraction_l529_52905

theorem greatest_fraction (a : ℝ) (m n p : ℝ) 
  (h_a : a < -3)
  (h_m : m = (a + 2) / (a + 3))
  (h_n : n = (a + 1) / (a + 2))
  (h_p : p = a / (a + 1)) :
  m > n ∧ n > p := by
sorry

end greatest_fraction_l529_52905


namespace travis_payment_l529_52956

def payment_calculation (total_bowls glass_bowls ceramic_bowls base_fee safe_delivery_fee
                         broken_glass_charge broken_ceramic_charge lost_glass_charge lost_ceramic_charge
                         additional_glass_fee additional_ceramic_fee lost_glass lost_ceramic
                         broken_glass broken_ceramic : ℕ) : ℚ :=
  let safe_glass := glass_bowls - lost_glass - broken_glass
  let safe_ceramic := ceramic_bowls - lost_ceramic - broken_ceramic
  let safe_delivery_payment := (safe_glass + safe_ceramic) * safe_delivery_fee
  let broken_lost_charges := broken_glass * broken_glass_charge + broken_ceramic * broken_ceramic_charge +
                             lost_glass * lost_glass_charge + lost_ceramic * lost_ceramic_charge
  let additional_moving_fee := glass_bowls * additional_glass_fee + ceramic_bowls * additional_ceramic_fee
  (base_fee + safe_delivery_payment - broken_lost_charges + additional_moving_fee : ℚ)

theorem travis_payment :
  payment_calculation 638 375 263 100 3 5 4 6 3 (1/2) (1/4) 9 3 10 5 = 2053.25 := by
  sorry

end travis_payment_l529_52956


namespace five_students_three_not_adjacent_l529_52945

/-- The number of ways to arrange n elements --/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange 5 students in a line --/
def totalArrangements : ℕ := factorial 5

/-- The number of ways to arrange 3 students together and 2 separately --/
def restrictedArrangements : ℕ := factorial 3 * factorial 3

/-- The number of ways to arrange 5 students where 3 are not adjacent --/
def validArrangements : ℕ := totalArrangements - restrictedArrangements

theorem five_students_three_not_adjacent :
  validArrangements = 84 :=
sorry

end five_students_three_not_adjacent_l529_52945


namespace prob_at_least_three_marbles_l529_52999

def num_green : ℕ := 5
def num_purple : ℕ := 7
def total_marbles : ℕ := num_green + num_purple
def num_draws : ℕ := 5

def prob_purple : ℚ := num_purple / total_marbles
def prob_green : ℚ := num_green / total_marbles

def prob_exactly (k : ℕ) : ℚ :=
  (Nat.choose num_draws k) * (prob_purple ^ k) * (prob_green ^ (num_draws - k))

def prob_at_least_three : ℚ :=
  prob_exactly 3 + prob_exactly 4 + prob_exactly 5

theorem prob_at_least_three_marbles :
  prob_at_least_three = 162582 / 2985984 := by
  sorry

end prob_at_least_three_marbles_l529_52999


namespace percent_of_double_is_nine_l529_52904

theorem percent_of_double_is_nine (x : ℝ) : 
  x > 0 → (0.01 * x * (2 * x) = 9) → x = 15 * Real.sqrt 2 := by
  sorry

end percent_of_double_is_nine_l529_52904


namespace tan_beta_value_l529_52951

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end tan_beta_value_l529_52951


namespace problem_solution_l529_52953

theorem problem_solution :
  ∀ (a b c d : ℝ),
    1000 * a = 85^2 - 15^2 →
    5 * a + 2 * b = 41 →
    (-3)^2 + 6 * (-3) + c = 0 →
    d^2 = (5 - c)^2 + (4 - 1)^2 →
    a = 7 ∧ b = 3 ∧ c = 9 ∧ d = 5 := by
  sorry

end problem_solution_l529_52953


namespace miriam_monday_pushups_l529_52941

/-- Represents the number of push-ups Miriam did on each day of the week --/
structure PushUps where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of push-ups done before Thursday --/
def totalBeforeThursday (p : PushUps) : ℕ :=
  p.monday + p.tuesday + p.wednesday

/-- Represents Miriam's push-up routine for the week --/
def miriamPushUps (monday : ℕ) : PushUps :=
  { monday := monday
  , tuesday := 7
  , wednesday := 2 * 7
  , thursday := (totalBeforeThursday { monday := monday, tuesday := 7, wednesday := 2 * 7, thursday := 0, friday := 0 }) / 2
  , friday := 39
  }

/-- Theorem stating that Miriam did 5 push-ups on Monday --/
theorem miriam_monday_pushups :
  ∃ (p : PushUps), p = miriamPushUps 5 ∧
    p.monday + p.tuesday + p.wednesday + p.thursday = p.friday :=
  sorry

end miriam_monday_pushups_l529_52941


namespace perfect_square_trinomial_l529_52985

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x y : ℝ, 16*x^2 + m*x*y + 25*y^2 = (a*x + b*y)^2) → 
  m = 40 ∨ m = -40 := by
sorry

end perfect_square_trinomial_l529_52985


namespace bianca_extra_flowers_l529_52980

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses daffodils sunflowers used : ℕ) : ℕ :=
  tulips + roses + daffodils + sunflowers - used

/-- Proof that Bianca picked 29 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 57 73 45 35 181 = 29 := by
  sorry

end bianca_extra_flowers_l529_52980


namespace tony_puzzle_time_l529_52965

/-- Calculates the total time spent solving puzzles given the time for a warm-up puzzle
    and the number and relative duration of additional puzzles. -/
def total_puzzle_time (warm_up_time : ℕ) (num_additional_puzzles : ℕ) (additional_puzzle_factor : ℕ) : ℕ :=
  warm_up_time + num_additional_puzzles * (warm_up_time * additional_puzzle_factor)

/-- Proves that given the specific conditions of Tony's puzzle-solving session,
    the total time spent is 70 minutes. -/
theorem tony_puzzle_time :
  total_puzzle_time 10 2 3 = 70 := by
  sorry

end tony_puzzle_time_l529_52965


namespace cos_150_degrees_l529_52943

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l529_52943


namespace program_output_l529_52955

theorem program_output : ∀ (a b : ℕ), a = 1 → b = 2 → a + b = 3 :=
by
  sorry

end program_output_l529_52955


namespace ernie_circles_l529_52972

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end ernie_circles_l529_52972


namespace brush_cost_is_correct_l529_52931

/-- The cost of a set of brushes for Maria's painting project -/
def brush_cost : ℝ := 20

/-- The cost of canvas for Maria's painting project -/
def canvas_cost : ℝ := 3 * brush_cost

/-- The cost of paint for Maria's painting project -/
def paint_cost : ℝ := 40

/-- The total cost of materials for Maria's painting project -/
def total_cost : ℝ := brush_cost + canvas_cost + paint_cost

/-- Theorem stating that the brush cost is correct given the problem conditions -/
theorem brush_cost_is_correct :
  brush_cost = 20 ∧
  canvas_cost = 3 * brush_cost ∧
  paint_cost = 40 ∧
  total_cost = 120 := by
  sorry

end brush_cost_is_correct_l529_52931


namespace chrome_users_l529_52940

theorem chrome_users (total : ℕ) (angle : ℕ) (chrome_users : ℕ) : 
  total = 530 → angle = 216 → chrome_users = 318 →
  (chrome_users : ℚ) / total * 360 = angle := by
  sorry

end chrome_users_l529_52940


namespace jeep_distance_calculation_l529_52974

theorem jeep_distance_calculation (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) :
  initial_time = 7 →
  speed = 40 →
  time_factor = 3 / 2 →
  (speed * (time_factor * initial_time)) = 420 :=
by sorry

end jeep_distance_calculation_l529_52974


namespace monica_reading_plan_l529_52927

def books_last_year : ℕ := 16

def books_this_year : ℕ := 2 * books_last_year

def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_reading_plan : books_next_year = 69 := by
  sorry

end monica_reading_plan_l529_52927


namespace complex_equation_solutions_l529_52903

theorem complex_equation_solutions :
  (∃ (x y : ℝ), (x + y) + (y - 1) * I = (2 * x + 3 * y) + (2 * y + 1) * I ∧ x = 4 ∧ y = -2) ∧
  (∃ (x y : ℝ), (x + y - 3) + (x - 2) * I = 0 ∧ x = 2 ∧ y = 1) :=
by sorry

end complex_equation_solutions_l529_52903


namespace expression_simplification_l529_52928

theorem expression_simplification (a : ℤ) (h : a = 2021) :
  (((a + 1 : ℚ) / a + 1 / (a + 1)) - a / (a + 1)) = (a^2 + a + 2 : ℚ) / (a * (a + 1)) ∧
  a^2 + a + 2 = 4094865 :=
sorry

end expression_simplification_l529_52928
