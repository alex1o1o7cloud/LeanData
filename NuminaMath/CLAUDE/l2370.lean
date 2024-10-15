import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_l2370_237029

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- First three terms increasing -/
def first_three_increasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three (a : ℕ → ℝ) :
  geometric_sequence a →
  (increasing_sequence a ↔ first_three_increasing a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_three_l2370_237029


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l2370_237005

/-- Represents a quadrilateral with side lengths and diagonal lengths. -/
structure Quadrilateral where
  a : ℝ  -- Length of side AB
  b : ℝ  -- Length of side BC
  c : ℝ  -- Length of side CD
  d : ℝ  -- Length of side DA
  m : ℝ  -- Length of diagonal AC
  n : ℝ  -- Length of diagonal BD
  A : ℝ  -- Angle at vertex A
  C : ℝ  -- Angle at vertex C

/-- Theorem stating the relationship between side lengths, diagonal lengths, and angles in a quadrilateral. -/
theorem quadrilateral_diagonal_theorem (q : Quadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_theorem_l2370_237005


namespace NUMINAMATH_CALUDE_rectangle_placement_l2370_237033

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (α : ℝ), a * (Real.cos α) + b * (Real.sin α) ≤ c ∧ 
              a * (Real.sin α) + b * (Real.cos α) ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_placement_l2370_237033


namespace NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2370_237041

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let time_in_hours := meeting_time / 60
  let circumference := relative_speed * time_in_hours
  circumference

/-- The actual problem statement -/
theorem jogging_track_circumference : 
  ∃ (c : ℝ), abs (c - track_circumference 20 17 37) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2370_237041


namespace NUMINAMATH_CALUDE_total_amount_calculation_l2370_237006

/-- The total amount Kanul had -/
def T : ℝ := sorry

/-- Theorem stating the relationship between the total amount and the expenses -/
theorem total_amount_calculation :
  T = 3000 + 2000 + 0.1 * T ∧ T = 5000 / 0.9 := by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l2370_237006


namespace NUMINAMATH_CALUDE_students_not_playing_l2370_237035

theorem students_not_playing (total : ℕ) (basketball : ℕ) (volleyball : ℕ) (both : ℕ) : 
  total = 20 ∧ 
  basketball = total / 2 ∧ 
  volleyball = total * 2 / 5 ∧ 
  both = total / 10 → 
  total - (basketball + volleyball - both) = 4 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_l2370_237035


namespace NUMINAMATH_CALUDE_investment_ratio_a_to_b_l2370_237089

/-- Given the investment ratios and profit distribution, prove the ratio of investments between A and B -/
theorem investment_ratio_a_to_b :
  ∀ (a b c total_investment total_profit : ℚ),
  -- A and C invested in ratio 3:2
  a / c = 3 / 2 →
  -- Total investment
  total_investment = a + b + c →
  -- Total profit
  total_profit = 60000 →
  -- C's profit
  c / total_investment * total_profit = 20000 →
  -- Prove that A:B = 3:1
  a / b = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_investment_ratio_a_to_b_l2370_237089


namespace NUMINAMATH_CALUDE_dot_product_zero_l2370_237065

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line on which P lies
def Line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_zero (P : ℝ × ℝ) (h : Line P.1 P.2) :
  dotProduct (P.1 - A.1, P.2 - A.2) (P.1 - B.1, P.2 - B.2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_l2370_237065


namespace NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l2370_237045

theorem x_cube_plus_reciprocal (φ : Real) (x : Real) 
  (h1 : 0 < φ) (h2 : φ < π) (h3 : x + 1/x = 2 * Real.cos (2 * φ)) : 
  x^3 + 1/x^3 = 2 * Real.cos (6 * φ) := by
  sorry

end NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l2370_237045


namespace NUMINAMATH_CALUDE_abs_seven_minus_sqrt_two_l2370_237064

theorem abs_seven_minus_sqrt_two (h : Real.sqrt 2 < 7) : 
  |7 - Real.sqrt 2| = 7 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_seven_minus_sqrt_two_l2370_237064


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l2370_237098

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) : ℝ) / x = 7 / 8 → x ≤ 48 / 7 := by
sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l2370_237098


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2370_237014

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}
def Q : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2370_237014


namespace NUMINAMATH_CALUDE_roots_of_equation_l2370_237093

theorem roots_of_equation (x : ℝ) :
  x * (x - 3)^2 * (5 + x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2370_237093


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inclusion_l2370_237059

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x + 2| + |x - 1| ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a_for_inclusion :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x + |x - 1| ≤ 2) → -3/2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_inclusion_l2370_237059


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2370_237032

theorem coefficient_x5_in_expansion :
  let n : ℕ := 36
  let k : ℕ := 5
  let coeff : ℤ := (n.choose k) * (-2 : ℤ) ^ (n - k)
  coeff = -8105545721856 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2370_237032


namespace NUMINAMATH_CALUDE_find_b_value_l2370_237096

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2370_237096


namespace NUMINAMATH_CALUDE_equation_solutions_l2370_237047

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 6 = 31 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, 1 - 8 * (1/4 + 0.5 * x) = 3 * (1 - 2 * x) ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2370_237047


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l2370_237086

def num_blue_balls : ℕ := 8
def num_yellow_balls : ℕ := 5
def total_balls : ℕ := num_blue_balls + num_yellow_balls

def prob_same_color : ℚ :=
  (num_blue_balls / total_balls) ^ 2 + (num_yellow_balls / total_balls) ^ 2

theorem prob_same_color_is_89_169 :
  prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l2370_237086


namespace NUMINAMATH_CALUDE_cloth_sold_meters_l2370_237016

/-- Proves that the number of meters of cloth sold is 80 -/
theorem cloth_sold_meters (total_selling_price : ℝ) (profit_per_meter : ℝ) (cost_price_per_meter : ℝ)
  (h1 : total_selling_price = 6900)
  (h2 : profit_per_meter = 20)
  (h3 : cost_price_per_meter = 66.25) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sold_meters_l2370_237016


namespace NUMINAMATH_CALUDE_circle_intersections_l2370_237009

-- Define the circle based on the given diameter endpoints
def circle_from_diameter (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius := ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt / 2
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the given circle
def given_circle : Set (ℝ × ℝ) := circle_from_diameter (2, 10) (14, 2)

-- Theorem statement
theorem circle_intersections :
  -- The x-coordinates of the intersections with the x-axis are 4 and 12
  (∃ (x : ℝ), (x, 0) ∈ given_circle ↔ x = 4 ∨ x = 12) ∧
  -- There are no intersections with the y-axis
  (∀ (y : ℝ), (0, y) ∉ given_circle) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersections_l2370_237009


namespace NUMINAMATH_CALUDE_ellipse_condition_l2370_237019

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (k : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1) ∧
  (2 - k > 0) ∧ (2*k - 1 > 0) ∧ (2 - k > 2*k - 1)

theorem ellipse_condition (k : ℝ) :
  is_ellipse_on_x_axis k ↔ 1/2 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2370_237019


namespace NUMINAMATH_CALUDE_sergio_fruit_sales_l2370_237085

/-- Calculates the total amount of money earned from fruit sales given the production of mangoes -/
def totalFruitSales (mangoProduction : ℕ) : ℕ :=
  let appleProduction := 2 * mangoProduction
  let orangeProduction := mangoProduction + 200
  let totalProduction := appleProduction + mangoProduction + orangeProduction
  totalProduction * 50

/-- Theorem stating that given the conditions, Mr. Sergio's total sales amount to $90,000 -/
theorem sergio_fruit_sales : totalFruitSales 400 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_sergio_fruit_sales_l2370_237085


namespace NUMINAMATH_CALUDE_jerrys_shelf_l2370_237077

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The initial number of action figures -/
def initial_action_figures : ℕ := 4

/-- The number of action figures added -/
def added_action_figures : ℕ := 2

/-- The difference between action figures and books -/
def difference : ℕ := 3

theorem jerrys_shelf :
  num_books = 3 ∧
  initial_action_figures + added_action_figures = num_books + difference :=
sorry

end NUMINAMATH_CALUDE_jerrys_shelf_l2370_237077


namespace NUMINAMATH_CALUDE_B_proper_subset_A_l2370_237087

-- Define sets A and B
def A : Set ℝ := {x | x > (1/2)}
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem B_proper_subset_A : B ⊂ A := by sorry

end NUMINAMATH_CALUDE_B_proper_subset_A_l2370_237087


namespace NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l2370_237001

/-- The vertex coordinates of the quadratic function y = 2x^2 - 4x + 5 are (1, 3) -/
theorem quadratic_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  ∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    (∀ x : ℝ, f x = 2 * (x - h)^2 + k) ∧
    (∀ x : ℝ, f x ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_coordinates_l2370_237001


namespace NUMINAMATH_CALUDE_license_plate_count_l2370_237036

/-- The number of possible letters in each letter position of the license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of the license plate. -/
def num_digits : ℕ := 10

/-- The number of letter positions in the license plate. -/
def num_letter_positions : ℕ := 3

/-- The number of digit positions in the license plate. -/
def num_digit_positions : ℕ := 4

/-- The total number of possible license plates in Eldorado. -/
def total_license_plates : ℕ := num_letters ^ num_letter_positions * num_digits ^ num_digit_positions

theorem license_plate_count :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l2370_237036


namespace NUMINAMATH_CALUDE_melanie_plums_theorem_l2370_237025

/-- Represents the number of plums Melanie has -/
def plums_remaining (initial_plums : ℕ) (plums_given_away : ℕ) : ℕ :=
  initial_plums - plums_given_away

/-- Theorem stating that Melanie's remaining plums are correctly calculated -/
theorem melanie_plums_theorem (initial_plums : ℕ) (plums_given_away : ℕ) 
  (h : initial_plums ≥ plums_given_away) :
  plums_remaining initial_plums plums_given_away = initial_plums - plums_given_away :=
by sorry

end NUMINAMATH_CALUDE_melanie_plums_theorem_l2370_237025


namespace NUMINAMATH_CALUDE_lost_shoes_count_l2370_237031

/-- Given an initial number of shoe pairs and a remaining number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 20 initial pairs and 15 remaining pairs,
    10 individual shoes are lost. -/
theorem lost_shoes_count : shoes_lost 20 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lost_shoes_count_l2370_237031


namespace NUMINAMATH_CALUDE_prob_white_then_yellow_is_two_thirds_l2370_237039

/-- The probability of drawing a white ball first, followed by a yellow ball, 
    from a bag containing 6 yellow and 4 white ping pong balls, 
    when drawing two balls without replacement. -/
def prob_white_then_yellow : ℚ :=
  let total_balls : ℕ := 10
  let yellow_balls : ℕ := 6
  let white_balls : ℕ := 4
  let prob_white_first : ℚ := white_balls / total_balls
  let prob_yellow_second : ℚ := yellow_balls / (total_balls - 1)
  prob_white_first * prob_yellow_second

theorem prob_white_then_yellow_is_two_thirds :
  prob_white_then_yellow = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_then_yellow_is_two_thirds_l2370_237039


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2370_237037

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 8) = 84 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l2370_237037


namespace NUMINAMATH_CALUDE_largest_red_socks_proof_l2370_237092

/-- The largest number of red socks satisfying the given conditions -/
def largest_red_socks : ℕ := 1164

/-- The total number of socks -/
def total_socks : ℕ := 1936

/-- Probability of selecting two socks of the same color -/
def same_color_prob : ℚ := 3/5

theorem largest_red_socks_proof :
  (total_socks ≤ 2500) ∧
  (largest_red_socks > (total_socks - largest_red_socks)) ∧
  (largest_red_socks * (largest_red_socks - 1) + 
   (total_socks - largest_red_socks) * (total_socks - largest_red_socks - 1)) / 
   (total_socks * (total_socks - 1)) = same_color_prob ∧
  (∀ r : ℕ, r > largest_red_socks → 
    (r ≤ total_socks ∧ r > (total_socks - r) ∧
     (r * (r - 1) + (total_socks - r) * (total_socks - r - 1)) / 
     (total_socks * (total_socks - 1)) = same_color_prob) → false) :=
by sorry

end NUMINAMATH_CALUDE_largest_red_socks_proof_l2370_237092


namespace NUMINAMATH_CALUDE_points_five_units_away_l2370_237027

theorem points_five_units_away (x : ℝ) : 
  (|x - 2| = 5) ↔ (x = 7 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_points_five_units_away_l2370_237027


namespace NUMINAMATH_CALUDE_smallest_cube_ending_584_l2370_237084

theorem smallest_cube_ending_584 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 584 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 584 [ZMOD 1000] → n ≤ m ∧ n = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_584_l2370_237084


namespace NUMINAMATH_CALUDE_time_per_braid_l2370_237073

/-- The time it takes to braid one braid, given the number of dancers, braids per dancer, and total time -/
theorem time_per_braid (num_dancers : ℕ) (braids_per_dancer : ℕ) (total_time_minutes : ℕ) : 
  num_dancers = 8 → 
  braids_per_dancer = 5 → 
  total_time_minutes = 20 → 
  (total_time_minutes * 60) / (num_dancers * braids_per_dancer) = 30 := by
  sorry

#check time_per_braid

end NUMINAMATH_CALUDE_time_per_braid_l2370_237073


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2370_237079

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2370_237079


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_35_l2370_237091

-- Define a function that represents angles with the same terminal side as a given angle
def sameTerminalSide (angle : ℝ) : ℤ → ℝ := fun k => k * 360 + angle

-- Theorem statement
theorem angle_with_same_terminal_side_as_negative_35 :
  ∃ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ ∃ (k : ℤ), x = sameTerminalSide (-35) k ∧ x = 325 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_35_l2370_237091


namespace NUMINAMATH_CALUDE_max_sin_x_value_l2370_237023

theorem max_sin_x_value (x y z : ℝ) 
  (h1 : Real.sin x = Real.cos y) 
  (h2 : Real.sin y = Real.cos z) 
  (h3 : Real.sin z = Real.cos x) : 
  ∃ (max_sin_x : ℝ), max_sin_x = Real.sqrt 2 / 2 ∧ 
    ∀ t, Real.sin t ≤ max_sin_x := by
  sorry

end NUMINAMATH_CALUDE_max_sin_x_value_l2370_237023


namespace NUMINAMATH_CALUDE_tangent_identities_l2370_237061

theorem tangent_identities :
  (∃ (f : ℝ → ℝ), (∀ x, f x = Real.tan x) ∧
    (f (π / 7) * f (2 * π / 7) * f (3 * π / 7) = Real.sqrt 7) ∧
    (f (π / 7)^2 + f (2 * π / 7)^2 + f (3 * π / 7)^2 = 21)) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_identities_l2370_237061


namespace NUMINAMATH_CALUDE_group_size_proof_l2370_237017

/-- The number of people in a group where:
    1. The total weight increase is 2.5 kg times the number of people.
    2. The weight difference between the new person and the replaced person is 20 kg. -/
def number_of_people : ℕ := 8

theorem group_size_proof :
  ∃ (n : ℕ), n = number_of_people ∧ 
  (2.5 : ℝ) * n = (20 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2370_237017


namespace NUMINAMATH_CALUDE_dollar_hash_composition_l2370_237012

def dollar (N : ℝ) : ℝ := 2 * (N + 1)

def hash (N : ℝ) : ℝ := 0.5 * N + 1

theorem dollar_hash_composition : hash (dollar (dollar (dollar 5))) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dollar_hash_composition_l2370_237012


namespace NUMINAMATH_CALUDE_F_6_indeterminate_l2370_237094

theorem F_6_indeterminate (F : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
  (h2 : ¬ F 7) :
  (F 6 ∨ ¬ F 6) :=
sorry

end NUMINAMATH_CALUDE_F_6_indeterminate_l2370_237094


namespace NUMINAMATH_CALUDE_ceiling_minus_y_is_half_l2370_237048

theorem ceiling_minus_y_is_half (x : ℝ) (y : ℝ) 
  (h1 : ⌈x⌉ - ⌊x⌋ = 0) 
  (h2 : y = x + 1/2) : 
  ⌈y⌉ - y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_is_half_l2370_237048


namespace NUMINAMATH_CALUDE_area_ratio_hexagon_octagon_l2370_237043

noncomputable def hexagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * (11 / 3) * side_length^2

noncomputable def octagon_circumradius (side_length : ℝ) : ℝ :=
  side_length * (2 * Real.sqrt 2) / Real.sqrt (2 - Real.sqrt 2)

noncomputable def octagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * ((octagon_circumradius side_length)^2 - (3 + 2 * Real.sqrt 2) * side_length^2)

theorem area_ratio_hexagon_octagon (side_length : ℝ) (h : side_length > 0) :
  hexagon_area_between_circles side_length / octagon_area_between_circles side_length =
  11 / (3 * ((octagon_circumradius 1)^2 - (3 + 2 * Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_hexagon_octagon_l2370_237043


namespace NUMINAMATH_CALUDE_complex_modulus_equal_parts_l2370_237099

theorem complex_modulus_equal_parts (b : ℝ) :
  let z : ℂ := (3 - b * Complex.I) / Complex.I
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_equal_parts_l2370_237099


namespace NUMINAMATH_CALUDE_total_distance_to_grandma_l2370_237071

/-- The distance to Grandma's house -/
def distance_to_grandma (distance_to_pie_shop : ℕ) (distance_to_gas_station : ℕ) (remaining_distance : ℕ) : ℕ :=
  distance_to_pie_shop + distance_to_gas_station + remaining_distance

/-- Theorem: The total distance to Grandma's house is 78 miles -/
theorem total_distance_to_grandma : 
  distance_to_grandma 35 18 25 = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_grandma_l2370_237071


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l2370_237026

/-- The fixed point of the function f(x) = a^(x-2) + 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) + 1
  f 2 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l2370_237026


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l2370_237066

theorem smallest_non_prime_non_square_no_small_factors : ∃ n : ℕ,
  n = 5183 ∧
  (∀ m : ℕ, m < n →
    (Nat.Prime m → m ≥ 70) ∧
    (¬ Nat.Prime n) ∧
    (∀ k : ℕ, k * k ≠ n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l2370_237066


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2370_237054

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2370_237054


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2370_237049

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l2370_237049


namespace NUMINAMATH_CALUDE_sam_win_probability_l2370_237007

/-- The probability of hitting the target with one shot -/
def hit_prob : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_prob : ℚ := 3/5

/-- The probability that Sam wins the game -/
def win_prob : ℚ := 5/8

theorem sam_win_probability :
  (hit_prob + miss_prob * miss_prob * win_prob = win_prob) →
  win_prob = 5/8 := by sorry

end NUMINAMATH_CALUDE_sam_win_probability_l2370_237007


namespace NUMINAMATH_CALUDE_graph_shift_l2370_237030

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the shift transformation
def shift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

-- Theorem statement
theorem graph_shift (a : ℝ) :
  ∀ x : ℝ, (shift g a) x = g (x - a) :=
by sorry

end NUMINAMATH_CALUDE_graph_shift_l2370_237030


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2370_237028

/-- Represents the number of students in each grade and the sample size for grade 10 -/
structure SchoolData where
  grade12 : ℕ
  grade11 : ℕ
  grade10 : ℕ
  sample10 : ℕ

/-- Calculates the total number of students sampled from the entire school using stratified sampling -/
def totalSampleSize (data : SchoolData) : ℕ :=
  (data.sample10 * (data.grade12 + data.grade11 + data.grade10)) / data.grade10

/-- Theorem stating that given the specific school data, the total sample size is 220 -/
theorem stratified_sample_size :
  let data := SchoolData.mk 700 700 800 80
  totalSampleSize data = 220 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2370_237028


namespace NUMINAMATH_CALUDE_gary_paycheck_l2370_237069

/-- Calculates the total paycheck for an employee with overtime --/
def calculate_paycheck (normal_wage : ℚ) (total_hours : ℕ) (regular_hours : ℕ) (overtime_multiplier : ℚ) : ℚ :=
  let regular_pay := normal_wage * regular_hours
  let overtime_hours := total_hours - regular_hours
  let overtime_pay := normal_wage * overtime_multiplier * overtime_hours
  regular_pay + overtime_pay

/-- Gary's paycheck calculation --/
theorem gary_paycheck :
  let normal_wage : ℚ := 12
  let total_hours : ℕ := 52
  let regular_hours : ℕ := 40
  let overtime_multiplier : ℚ := 3/2
  calculate_paycheck normal_wage total_hours regular_hours overtime_multiplier = 696 := by
  sorry


end NUMINAMATH_CALUDE_gary_paycheck_l2370_237069


namespace NUMINAMATH_CALUDE_age_difference_is_six_l2370_237072

-- Define Claire's future age
def claire_future_age : ℕ := 20

-- Define the number of years until Claire reaches her future age
def years_until_future : ℕ := 2

-- Define Jessica's current age
def jessica_current_age : ℕ := 24

-- Theorem to prove
theorem age_difference_is_six :
  jessica_current_age - (claire_future_age - years_until_future) = 6 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_is_six_l2370_237072


namespace NUMINAMATH_CALUDE_sean_houses_problem_l2370_237063

theorem sean_houses_problem (initial_houses : ℕ) : 
  initial_houses - 8 + 12 = 31 → initial_houses = 27 := by
  sorry

end NUMINAMATH_CALUDE_sean_houses_problem_l2370_237063


namespace NUMINAMATH_CALUDE_charlie_banana_consumption_l2370_237053

/-- Represents the daily banana consumption of Charlie the chimp over 7 days -/
def BananaSequence : Type := Fin 7 → ℚ

/-- The sum of bananas eaten over 7 days is 150 -/
def SumIs150 (seq : BananaSequence) : Prop :=
  (Finset.sum Finset.univ seq) = 150

/-- Each day's consumption is 4 more than the previous day -/
def ArithmeticProgression (seq : BananaSequence) : Prop :=
  ∀ i : Fin 6, seq (i.succ) = seq i + 4

/-- The theorem to be proved -/
theorem charlie_banana_consumption
  (seq : BananaSequence)
  (sum_cond : SumIs150 seq)
  (prog_cond : ArithmeticProgression seq) :
  seq 6 = 33 + 4/7 := by sorry

end NUMINAMATH_CALUDE_charlie_banana_consumption_l2370_237053


namespace NUMINAMATH_CALUDE_snowflake_area_ratio_l2370_237000

/-- Represents the snowflake shape after n iterations --/
def Snowflake (n : ℕ) : Type := Unit

/-- The area of the snowflake shape after n iterations --/
def area (s : Snowflake n) : ℚ := sorry

/-- The initial equilateral triangle --/
def initial_triangle : Snowflake 0 := sorry

/-- The snowflake shape after one iteration --/
def first_iteration : Snowflake 1 := sorry

/-- The snowflake shape after two iterations --/
def second_iteration : Snowflake 2 := sorry

theorem snowflake_area_ratio :
  area second_iteration / area initial_triangle = 40 / 27 := by sorry

end NUMINAMATH_CALUDE_snowflake_area_ratio_l2370_237000


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2370_237083

/-- The lateral area of a cylinder with base diameter and height both 4 cm is 16π cm² -/
theorem cylinder_lateral_area (π : ℝ) : 
  let base_diameter : ℝ := 4
  let height : ℝ := 4
  let lateral_area : ℝ := π * base_diameter * height
  lateral_area = 16 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2370_237083


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2370_237050

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCardCount : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

/-- The expected number of pairs of adjacent red cards in a standard 52-card deck
    dealt in a circle -/
theorem expected_adjacent_red_pairs :
  (redCardCount : ℚ) * probAdjacentRed = 650 / 51 := by sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2370_237050


namespace NUMINAMATH_CALUDE_expanded_ohara_triple_solution_l2370_237008

/-- An Expanded O'Hara triple is a tuple of four positive integers (a, b, c, x) 
    such that √a + √b + √c = x -/
def IsExpandedOHaraTriple (a b c x : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b + Real.sqrt c = x

theorem expanded_ohara_triple_solution :
  IsExpandedOHaraTriple 49 64 16 19 := by sorry

end NUMINAMATH_CALUDE_expanded_ohara_triple_solution_l2370_237008


namespace NUMINAMATH_CALUDE_lens_circumference_approx_l2370_237090

-- Define π as a constant (approximation)
def π : ℝ := 3.14159

-- Define the diameter of the lens
def d : ℝ := 10

-- Define the circumference calculation function
def circumference (diameter : ℝ) : ℝ := π * diameter

-- Theorem statement
theorem lens_circumference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |circumference d - 31.42| < ε :=
sorry

end NUMINAMATH_CALUDE_lens_circumference_approx_l2370_237090


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2370_237011

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum : a 1 + a 2 + a 3 = 12)
  (h_prod : a 1 * a 2 * a 3 = 48) :
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2370_237011


namespace NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l2370_237074

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum equals 6) when throwing two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of the sum of two fair dice equaling 6 -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths : 
  prob_sum_six = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_six_is_five_thirty_sixths_l2370_237074


namespace NUMINAMATH_CALUDE_flat_fee_is_40_l2370_237018

/-- A hotel pricing structure with a flat fee for the first night and a fixed amount for each additional night. -/
structure HotelPricing where
  flatFee : ℝ
  additionalNightFee : ℝ

/-- Calculate the total cost for a stay given the pricing structure and number of nights. -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.additionalNightFee * (nights - 1)

/-- The flat fee for the first night is $40 given the conditions. -/
theorem flat_fee_is_40 :
  ∃ (pricing : HotelPricing),
    totalCost pricing 4 = 195 ∧
    totalCost pricing 7 = 350 ∧
    pricing.flatFee = 40 := by
  sorry

end NUMINAMATH_CALUDE_flat_fee_is_40_l2370_237018


namespace NUMINAMATH_CALUDE_custom_mult_eleven_twelve_l2370_237024

/-- Custom multiplication operation for integers -/
def custom_mult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that for y = 11, y * 12 = 110 under the custom multiplication -/
theorem custom_mult_eleven_twelve :
  let y : ℤ := 11
  custom_mult y 12 = 110 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_eleven_twelve_l2370_237024


namespace NUMINAMATH_CALUDE_complex_modulus_l2370_237082

theorem complex_modulus (z : ℂ) : (1 + Complex.I * Real.sqrt 3) * z = 1 + Complex.I →
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2370_237082


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l2370_237080

theorem binomial_coefficient_equation_solution : 
  ∃! n : ℕ, (Nat.choose 25 n) + (Nat.choose 25 12) = (Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_solution_l2370_237080


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l2370_237051

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs (white black red : ℕ) 
  (h_white : white = 5) 
  (h_black : black = 4) 
  (h_red : red = 3) : 
  (choose white 2) + (choose black 2) + (choose red 2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l2370_237051


namespace NUMINAMATH_CALUDE_power_3_2048_mod_11_l2370_237095

theorem power_3_2048_mod_11 : 3^2048 ≡ 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_power_3_2048_mod_11_l2370_237095


namespace NUMINAMATH_CALUDE_jims_out_of_pocket_l2370_237010

/-- The cost of Jim's first wedding ring in dollars -/
def first_ring_cost : ℕ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def second_ring_cost : ℕ := 2 * first_ring_cost

/-- The selling price of Jim's first ring in dollars -/
def first_ring_selling_price : ℕ := first_ring_cost / 2

/-- Jim's total out-of-pocket expense in dollars -/
def total_out_of_pocket : ℕ := second_ring_cost + (first_ring_cost - first_ring_selling_price)

/-- Theorem stating Jim's total out-of-pocket expense -/
theorem jims_out_of_pocket : total_out_of_pocket = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jims_out_of_pocket_l2370_237010


namespace NUMINAMATH_CALUDE_water_in_bucket_l2370_237020

/-- 
Given a bucket with an initial amount of water and an additional amount added,
calculate the total amount of water in the bucket.
-/
theorem water_in_bucket (initial : ℝ) (added : ℝ) :
  initial = 3 → added = 6.8 → initial + added = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l2370_237020


namespace NUMINAMATH_CALUDE_square_difference_cubed_l2370_237021

theorem square_difference_cubed : (5^2 - 4^2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_cubed_l2370_237021


namespace NUMINAMATH_CALUDE_exists_all_berries_l2370_237046

/-- A binary vector of length 7 -/
def BinaryVector := Fin 7 → Bool

/-- The set of 16 vectors representing the work schedule -/
def WorkSchedule := Fin 16 → BinaryVector

/-- The condition that the first vector is all zeros -/
def firstDayAllMine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule 0 i = false

/-- The condition that any two vectors differ in at least 3 positions -/
def atLeastThreeDifferences (schedule : WorkSchedule) : Prop :=
  ∀ d1 d2 : Fin 16, d1 ≠ d2 →
    (Finset.filter (fun i => schedule d1 i ≠ schedule d2 i) Finset.univ).card ≥ 3

/-- The theorem to be proved -/
theorem exists_all_berries (schedule : WorkSchedule)
  (h1 : firstDayAllMine schedule)
  (h2 : atLeastThreeDifferences schedule) :
  ∃ d : Fin 16, ∀ i : Fin 7, schedule d i = true := by
  sorry

end NUMINAMATH_CALUDE_exists_all_berries_l2370_237046


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2370_237081

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2370_237081


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2370_237068

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) 
  (sum_18 : n.hundreds + n.tens + n.units = 18)
  (hundreds_tens_relation : n.hundreds = n.tens + 1)
  (units_tens_relation : n.units = n.tens + 2) :
  n.toNat = 657 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2370_237068


namespace NUMINAMATH_CALUDE_solid_surface_area_l2370_237097

/-- The surface area of a solid composed of a cylinder topped with a hemisphere -/
theorem solid_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 3) :
  2 * π * r * h + 2 * π * r^2 + 2 * π * r^2 = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_solid_surface_area_l2370_237097


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2370_237034

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos ((2 * π) / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2370_237034


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2370_237088

theorem quadratic_rewrite (x : ℝ) : ∃ (a b c : ℤ), 
  16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c ∧ a * b = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2370_237088


namespace NUMINAMATH_CALUDE_option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l2370_237042

-- Define variables
variable (a b c : ℝ)

-- Theorem for Option A
theorem option_a_correct : a = b → a + 6 = b + 6 := by sorry

-- Theorem for Option B
theorem option_b_correct : a = b → a / 9 = b / 9 := by sorry

-- Theorem for Option C
theorem option_c_correct (h : c ≠ 0) : a / c = b / c → a = b := by sorry

-- Theorem for Option D (incorrect transformation)
theorem option_d_incorrect : ∃ a b : ℝ, -2 * a = -2 * b ∧ a ≠ -b := by sorry

end NUMINAMATH_CALUDE_option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l2370_237042


namespace NUMINAMATH_CALUDE_max_m_value_l2370_237070

-- Define the circle M
def circle_M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 5 = m}

-- Define points A and B
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the property of right angle APB
def is_right_angle (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - point_A.1, P.2 - point_A.2)
  let BP := (P.1 - point_B.1, P.2 - point_B.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- Theorem statement
theorem max_m_value :
  ∃ (m : ℝ), ∀ (m' : ℝ),
    (∃ (P : ℝ × ℝ), P ∈ circle_M m' ∧ is_right_angle P) →
    m' ≤ m ∧
    m = 45 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2370_237070


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2370_237058

theorem divisibility_by_five : ∃ k : ℤ, 3^444 + 4^333 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2370_237058


namespace NUMINAMATH_CALUDE_expression_evaluation_l2370_237055

theorem expression_evaluation : 2 + 3 * 4 - 5 * 6 + 7 = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2370_237055


namespace NUMINAMATH_CALUDE_product_of_sums_evaluate_specific_product_l2370_237062

theorem product_of_sums (a b : ℕ) : (a + 1) * (a^2 + 1^2) * (a^4 + 1^4) = ((a^2 - 1^2) * (a^2 + 1^2) * (a^4 - 1^4) * (a^4 + 1^4)) / (a - 1) / 2 := by
  sorry

theorem evaluate_specific_product : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_evaluate_specific_product_l2370_237062


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2370_237057

/-- The surface area of a rectangular box -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * h + l * w + w * h)

/-- Theorem: The surface area of a rectangular box with length l, width w, and height h
    is equal to 2(lh + lw + wh) -/
theorem rectangular_box_surface_area (l w h : ℝ) :
  surface_area l w h = 2 * (l * h + l * w + w * h) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2370_237057


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l2370_237004

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 12 / (x - 3) = 5 - 12 / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l2370_237004


namespace NUMINAMATH_CALUDE_same_terminal_side_l2370_237075

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def SameTerminalSide (a b : ℝ) : Prop := ∃ k : ℤ, a - b = k * 360

/-- The angle -510 degrees -/
def angle1 : ℝ := -510

/-- The angle 210 degrees -/
def angle2 : ℝ := 210

/-- Theorem: angle1 and angle2 have the same terminal side -/
theorem same_terminal_side : SameTerminalSide angle1 angle2 := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l2370_237075


namespace NUMINAMATH_CALUDE_square_condition_l2370_237013

def a_n (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem square_condition (n : ℕ+) (b : ℕ) : 
  0 < b ∧ b < 10 →
  (∃ k : ℕ, a_n (2*n) - b * a_n n = k^2) ↔ (b = 2 ∨ (b = 7 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_condition_l2370_237013


namespace NUMINAMATH_CALUDE_cylinder_triple_volume_radius_l2370_237076

/-- Theorem: Tripling the volume of a cylinder while keeping the same height results in a new radius that is √3 times the original radius. -/
theorem cylinder_triple_volume_radius (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let v := π * r^2 * h
  let v_new := 3 * v
  let r_new := Real.sqrt ((3 * π * r^2 * h) / (π * h))
  r_new = r * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_triple_volume_radius_l2370_237076


namespace NUMINAMATH_CALUDE_waiter_tips_ratio_l2370_237067

theorem waiter_tips_ratio (salary tips : ℝ) 
  (h : tips / (salary + tips) = 0.7142857142857143) : 
  tips / salary = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_ratio_l2370_237067


namespace NUMINAMATH_CALUDE_calculation_proof_l2370_237078

theorem calculation_proof : 0.54 - (1/8 : ℚ) + 0.46 - (7/8 : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2370_237078


namespace NUMINAMATH_CALUDE_unique_k_divisibility_l2370_237060

theorem unique_k_divisibility (a b l : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^l) :
  ∀ k : ℕ, k > 0 → (k^2 ∣ a^k + b^k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_divisibility_l2370_237060


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_45_l2370_237015

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem three_digit_number_divisible_by_45 (n : ℕ) :
  is_three_digit n →
  n % 45 = 0 →
  n - reverse_number n = 297 →
  n = 360 ∨ n = 855 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_45_l2370_237015


namespace NUMINAMATH_CALUDE_product_expansion_l2370_237052

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x^2) + 5*x - 6) = 6 / x^2 + (15*x) / 4 - 4.5 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2370_237052


namespace NUMINAMATH_CALUDE_range_of_function_l2370_237040

theorem range_of_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^k
  Set.range (fun x ↦ f x) = Set.Ici (3 * 2^k) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2370_237040


namespace NUMINAMATH_CALUDE_max_area_between_parabolas_l2370_237003

/-- The parabola C_a -/
def C_a (a x : ℝ) : ℝ := -2 * x^2 + 4 * a * x - 2 * a^2 + a + 1

/-- The parabola C -/
def C (x : ℝ) : ℝ := x^2 - 2 * x

/-- The difference function between C and C_a -/
def f_a (a x : ℝ) : ℝ := C x - C_a a x

/-- Theorem: The maximum area enclosed by parabolas C_a and C is 27/(4√2) -/
theorem max_area_between_parabolas :
  ∃ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_a a x₁ = 0 ∧ f_a a x₂ = 0) →
  (∫ (x : ℝ) in Set.Icc (min x₁ x₂) (max x₁ x₂), f_a a x) ≤ 27 / (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_area_between_parabolas_l2370_237003


namespace NUMINAMATH_CALUDE_custom_operation_equation_l2370_237022

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a + 2 * b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℝ, star 3 (star 4 x) = 6 ∧ x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equation_l2370_237022


namespace NUMINAMATH_CALUDE_max_product_value_l2370_237044

-- Define the functions h and k on ℝ
variable (h k : ℝ → ℝ)

-- Define the ranges of h and k
variable (h_range : Set.range h = Set.Icc (-3) 5)
variable (k_range : Set.range k = Set.Icc (-1) 3)

-- Theorem statement
theorem max_product_value :
  ∃ (x : ℝ), h x * k x = 15 ∧ ∀ (y : ℝ), h y * k y ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_product_value_l2370_237044


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2370_237056

def numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]

theorem mean_of_remaining_numbers :
  ∀ (four_nums : List ℕ),
    four_nums.length = 4 →
    four_nums.all (· ∈ numbers) →
    (four_nums.sum : ℚ) / 4 = 2008 →
    let remaining_nums := numbers.filter (· ∉ four_nums)
    (remaining_nums.sum : ℚ) / 2 = 2051 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l2370_237056


namespace NUMINAMATH_CALUDE_acrobat_count_range_l2370_237038

/-- Represents the count of animals in the zoo --/
structure AnimalCount where
  elephants : ℕ
  monkeys : ℕ
  acrobats : ℕ

/-- Checks if the animal count satisfies the given conditions --/
def isValidCount (count : AnimalCount) : Prop :=
  count.elephants * 4 + count.monkeys * 2 + count.acrobats * 2 = 50 ∧
  count.elephants + count.monkeys + count.acrobats = 18

/-- The main theorem stating the range of possible acrobat counts --/
theorem acrobat_count_range :
  ∀ n : ℕ, 0 ≤ n ∧ n ≤ 11 →
  ∃ (count : AnimalCount), isValidCount count ∧ count.acrobats = n :=
by sorry

end NUMINAMATH_CALUDE_acrobat_count_range_l2370_237038


namespace NUMINAMATH_CALUDE_minimum_matches_theorem_l2370_237002

/-- Represents the number of points for each match result -/
structure PointSystem where
  win : Nat
  draw : Nat
  loss : Nat

/-- Represents the state of a team in the competition -/
structure TeamState where
  gamesPlayed : Nat
  points : Nat

/-- Represents the requirements for the team -/
structure TeamRequirement where
  targetPoints : Nat
  minWinsNeeded : Nat

def minimumTotalMatches (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) : Nat :=
  initialState.gamesPlayed + requirement.minWinsNeeded +
    ((requirement.targetPoints - initialState.points - requirement.minWinsNeeded * pointSystem.win + pointSystem.draw - 1) / pointSystem.draw)

theorem minimum_matches_theorem (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) :
  initialState.gamesPlayed = 5 ∧
  initialState.points = 14 ∧
  pointSystem.win = 3 ∧
  pointSystem.draw = 1 ∧
  pointSystem.loss = 0 ∧
  requirement.targetPoints = 40 ∧
  requirement.minWinsNeeded = 6 →
  minimumTotalMatches initialState pointSystem requirement = 13 := by
  sorry

end NUMINAMATH_CALUDE_minimum_matches_theorem_l2370_237002
