import Mathlib

namespace NUMINAMATH_CALUDE_derek_same_color_probability_l3651_365142

/-- Represents the number of marbles of each color -/
structure MarbleDistribution :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Represents the number of marbles drawn by each person -/
structure DrawingProcess :=
  (david : ℕ)
  (dana : ℕ)
  (derek : ℕ)

/-- Calculates the probability of Derek getting at least 2 marbles of the same color -/
def probability_same_color (dist : MarbleDistribution) (process : DrawingProcess) : ℚ :=
  sorry

theorem derek_same_color_probability :
  let initial_distribution : MarbleDistribution := ⟨3, 2, 3⟩
  let drawing_process : DrawingProcess := ⟨2, 2, 3⟩
  probability_same_color initial_distribution drawing_process = 19 / 210 :=
sorry

end NUMINAMATH_CALUDE_derek_same_color_probability_l3651_365142


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3651_365146

theorem solve_exponential_equation :
  ∃ y : ℝ, (40 : ℝ)^3 = 8^y ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3651_365146


namespace NUMINAMATH_CALUDE_spice_difference_total_l3651_365131

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.35

theorem spice_difference_total : 
  abs (cinnamon - nutmeg) + abs (nutmeg - ginger) + abs (cinnamon - ginger) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_spice_difference_total_l3651_365131


namespace NUMINAMATH_CALUDE_download_rate_proof_l3651_365155

/-- Proves that the download rate for the first 60 megabytes is 5 megabytes per second -/
theorem download_rate_proof (file_size : ℝ) (first_part_size : ℝ) (second_part_rate : ℝ) (total_time : ℝ)
  (h1 : file_size = 90)
  (h2 : first_part_size = 60)
  (h3 : second_part_rate = 10)
  (h4 : total_time = 15)
  (h5 : file_size = first_part_size + (file_size - first_part_size))
  (h6 : total_time = first_part_size / R + (file_size - first_part_size) / second_part_rate) :
  R = 5 := by
  sorry

#check download_rate_proof

end NUMINAMATH_CALUDE_download_rate_proof_l3651_365155


namespace NUMINAMATH_CALUDE_card_width_is_15_l3651_365110

/-- A rectangular card with a given perimeter and width-length relationship -/
structure RectangularCard where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 46
  width_length_rel : width = length + 7

/-- The width of the rectangular card is 15 cm -/
theorem card_width_is_15 (card : RectangularCard) : card.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_card_width_is_15_l3651_365110


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l3651_365133

theorem division_multiplication_equality : (180 / 6) * 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l3651_365133


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l3651_365150

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
def applySimiliarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A does not belong to the image of plane a under the similarity transformation -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -6 }
  let a : Plane := { a := 6, b := -1, c := -1, d := 7 }
  let k : ℝ := 2/7
  let a_transformed := applySimiliarity a k
  ¬ satisfiesPlane A a_transformed :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l3651_365150


namespace NUMINAMATH_CALUDE_product_upper_bound_l3651_365171

theorem product_upper_bound (x : ℝ) (h : x ∈ Set.Icc 0 1) : x * (1 - x) ≤ (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_upper_bound_l3651_365171


namespace NUMINAMATH_CALUDE_total_passengers_in_hour_l3651_365193

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
def total_passengers (train_frequency : ℕ) (passengers_leaving : ℕ) (passengers_boarding : ℕ) : ℕ :=
  let trains_per_hour := 60 / train_frequency
  let passengers_per_train := passengers_leaving + passengers_boarding
  trains_per_hour * passengers_per_train

/-- Proves that given the specified conditions, the total number of different passengers in an hour is 6240 -/
theorem total_passengers_in_hour :
  total_passengers 5 200 320 = 6240 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_in_hour_l3651_365193


namespace NUMINAMATH_CALUDE_white_balls_count_l3651_365116

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  green = 18 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  prob = (total - (red + purple)) / total →
  total - (green + yellow + red + purple) = 21 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l3651_365116


namespace NUMINAMATH_CALUDE_cross_country_race_winning_scores_l3651_365162

/-- Represents a cross-country race with two teams -/
structure CrossCountryRace where
  /-- The number of players in each team -/
  players_per_team : Nat
  /-- The total number of players in the race -/
  total_players : Nat
  /-- The sum of all possible scores in the race -/
  total_score : Nat

/-- Calculates the maximum possible score for the winning team -/
def max_winning_score (race : CrossCountryRace) : Nat :=
  race.total_score / 2

/-- Calculates the minimum possible score for any team -/
def min_team_score (race : CrossCountryRace) : Nat :=
  List.sum (List.range race.players_per_team)

/-- The number of possible scores for the winning team -/
def winning_score_count (race : CrossCountryRace) : Nat :=
  max_winning_score race - min_team_score race + 1

/-- Theorem stating the number of possible scores for the winning team in a specific cross-country race -/
theorem cross_country_race_winning_scores :
  ∃ (race : CrossCountryRace),
    race.players_per_team = 5 ∧
    race.total_players = 10 ∧
    race.total_score = (race.total_players * (race.total_players + 1)) / 2 ∧
    winning_score_count race = 13 := by
  sorry


end NUMINAMATH_CALUDE_cross_country_race_winning_scores_l3651_365162


namespace NUMINAMATH_CALUDE_fraction_equality_l3651_365106

theorem fraction_equality (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -5 ∧ x ≠ 2 → 
    (A / (x + 2) + B / (x^2 - 4*x - 5) = (x^2 + x + 7) / (x^3 + 6*x^2 - 13*x - 10))) → 
  B / A = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3651_365106


namespace NUMINAMATH_CALUDE_binomial_constant_term_l3651_365102

/-- The constant term in the binomial expansion of (x - a/(3x))^8 -/
def constantTerm (a : ℝ) : ℝ := ((-1)^6 * a^6) * (Nat.choose 8 6)

theorem binomial_constant_term (a : ℝ) : 
  constantTerm a = 28 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_constant_term_l3651_365102


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3651_365153

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3651_365153


namespace NUMINAMATH_CALUDE_circles_configuration_l3651_365189

/-- Two circles with radii r₁ and r₂, and distance d between their centers,
    are in the "one circle inside the other" configuration if d < |r₁ - r₂| -/
def CircleInsideOther (r₁ r₂ d : ℝ) : Prop :=
  d < |r₁ - r₂|

/-- Given two circles with radii 1 and 5, and distance 3 between their centers,
    prove that one circle is inside the other -/
theorem circles_configuration :
  CircleInsideOther 1 5 3 := by
sorry

end NUMINAMATH_CALUDE_circles_configuration_l3651_365189


namespace NUMINAMATH_CALUDE_shorter_worm_length_l3651_365195

/-- Given two worms where one is 0.8 inches long and the other is 0.7 inches longer,
    prove that the length of the shorter worm is 0.8 inches. -/
theorem shorter_worm_length (worm1 worm2 : ℝ) 
  (h1 : worm1 = 0.8)
  (h2 : worm2 = worm1 + 0.7) :
  min worm1 worm2 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_shorter_worm_length_l3651_365195


namespace NUMINAMATH_CALUDE_lunks_needed_for_apples_l3651_365139

-- Define the exchange rates
def lunks_to_kunks (l : ℚ) : ℚ := l * (2/4)
def kunks_to_apples (k : ℚ) : ℚ := k * (5/3)

-- Theorem statement
theorem lunks_needed_for_apples (n : ℚ) : 
  kunks_to_apples (lunks_to_kunks 18) = 15 := by
  sorry

#check lunks_needed_for_apples

end NUMINAMATH_CALUDE_lunks_needed_for_apples_l3651_365139


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3651_365144

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (2 * x - 1 : ℂ) + i = y - (3 - y) * i →
  x = 5 / 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3651_365144


namespace NUMINAMATH_CALUDE_inverse_proportion_points_relation_l3651_365184

theorem inverse_proportion_points_relation :
  ∀ x₁ x₂ x₃ : ℝ,
  (2 = 8 / x₁) →
  (-1 = 8 / x₂) →
  (4 = 8 / x₃) →
  (x₁ > x₃ ∧ x₃ > x₂) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_relation_l3651_365184


namespace NUMINAMATH_CALUDE_concert_attendance_l3651_365188

theorem concert_attendance (adult_price child_price total_collected : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 3)
  (h3 : total_collected = 6000)
  (h4 : ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected) :
  ∃ (total : ℕ), total = 1500 ∧ 
    ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected ∧ 
    total = a + c := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3651_365188


namespace NUMINAMATH_CALUDE_second_day_hours_proof_l3651_365158

/-- Represents the number of hours worked on the second day -/
def hours_second_day : ℕ := 8

/-- The hourly rate paid to each worker -/
def hourly_rate : ℕ := 10

/-- The total payment received by both workers -/
def total_payment : ℕ := 660

/-- The number of hours worked on the first day -/
def hours_first_day : ℕ := 10

/-- The number of hours worked on the third day -/
def hours_third_day : ℕ := 15

/-- The number of workers -/
def num_workers : ℕ := 2

theorem second_day_hours_proof :
  hours_second_day * num_workers * hourly_rate +
  hours_first_day * num_workers * hourly_rate +
  hours_third_day * num_workers * hourly_rate = total_payment :=
by sorry

end NUMINAMATH_CALUDE_second_day_hours_proof_l3651_365158


namespace NUMINAMATH_CALUDE_sqrt_of_square_neg_l3651_365170

theorem sqrt_of_square_neg (a : ℝ) (h : a < 0) : Real.sqrt (a ^ 2) = -a := by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_neg_l3651_365170


namespace NUMINAMATH_CALUDE_problem_solution_l3651_365197

def f (x a : ℝ) := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) (1/2)) ∧
  (Set.Icc (1/2) 1 ⊆ {x : ℝ | f x a ≤ |2*x + 1|} → a ∈ Set.Icc 0 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3651_365197


namespace NUMINAMATH_CALUDE_factor_quadratic_l3651_365135

theorem factor_quadratic (x t : ℝ) : 
  (x - t) ∣ (10 * x^2 + 23 * x - 7) ↔ 
  t = (-23 + Real.sqrt 809) / 20 ∨ t = (-23 - Real.sqrt 809) / 20 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l3651_365135


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l3651_365156

/-- Given that 40 cows eat 40 bags of husk in 40 days, 
    prove that one cow will eat one bag of husk in 40 days. -/
theorem cow_husk_consumption (cows bags days : ℕ) 
  (h : cows = 40 ∧ bags = 40 ∧ days = 40) : 
  (cows * bags) / (cows * days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l3651_365156


namespace NUMINAMATH_CALUDE_census_suitability_l3651_365174

/-- Represents a survey --/
structure Survey where
  description : String
  population_size : Nat
  ease_of_survey : Bool

/-- Defines when a survey is suitable for a census --/
def suitable_for_census (s : Survey) : Prop :=
  s.population_size < 1000 ∧ s.ease_of_survey

/-- Theorem stating the condition for a survey to be suitable for a census --/
theorem census_suitability (s : Survey) :
  suitable_for_census s ↔ s.population_size < 1000 ∧ s.ease_of_survey := by sorry

end NUMINAMATH_CALUDE_census_suitability_l3651_365174


namespace NUMINAMATH_CALUDE_matching_probability_theorem_l3651_365167

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.yellow + jb.red

/-- Abe's jelly beans -/
def abe : JellyBeans :=
  { green := 2, yellow := 0, red := 3 }

/-- Bob's jelly beans -/
def bob : JellyBeans :=
  { green := 2, yellow := 2, red := 3 }

/-- Calculates the probability of matching colors -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  (person1.green * person2.green + person1.red * person2.red : ℚ) /
  ((person1.total * person2.total) : ℚ)

theorem matching_probability_theorem :
  matchingProbability abe bob = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_theorem_l3651_365167


namespace NUMINAMATH_CALUDE_linear_function_m_range_l3651_365198

/-- A linear function y = (m-1)x + (4m-3) whose graph lies in the first, second, and fourth quadrants -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + (4 * m - 3)

/-- The slope of the linear function is negative -/
def slope_negative (m : ℝ) : Prop := m - 1 < 0

/-- The y-intercept of the linear function is positive -/
def y_intercept_positive (m : ℝ) : Prop := 4 * m - 3 > 0

/-- The graph of the linear function lies in the first, second, and fourth quadrants -/
def graph_in_first_second_fourth_quadrants (m : ℝ) : Prop :=
  slope_negative m ∧ y_intercept_positive m

theorem linear_function_m_range :
  ∀ m : ℝ, graph_in_first_second_fourth_quadrants m → 3/4 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_m_range_l3651_365198


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3651_365143

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3651_365143


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3651_365124

theorem simplify_trig_expression (x : ℝ) :
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3651_365124


namespace NUMINAMATH_CALUDE_f_comparison_and_max_value_l3651_365117

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem f_comparison_and_max_value :
  (f (π / 4) > f (π / 6)) ∧
  (∀ x : ℝ, f x ≤ -3/2) ∧
  (∃ x : ℝ, f x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_comparison_and_max_value_l3651_365117


namespace NUMINAMATH_CALUDE_self_inverse_cube_mod_15_l3651_365140

theorem self_inverse_cube_mod_15 (a : ℤ) (h : a * a ≡ 1 [ZMOD 15]) :
  a^3 ≡ 1 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_cube_mod_15_l3651_365140


namespace NUMINAMATH_CALUDE_lisa_packing_peanuts_l3651_365192

/-- The amount of packing peanuts needed for a large order in grams -/
def large_order_peanuts : ℕ := 200

/-- The amount of packing peanuts needed for a small order in grams -/
def small_order_peanuts : ℕ := 50

/-- The number of large orders Lisa has sent -/
def large_orders : ℕ := 3

/-- The number of small orders Lisa has sent -/
def small_orders : ℕ := 4

/-- The total amount of packing peanuts used by Lisa -/
def total_peanuts : ℕ := large_order_peanuts * large_orders + small_order_peanuts * small_orders

theorem lisa_packing_peanuts : total_peanuts = 800 := by
  sorry

end NUMINAMATH_CALUDE_lisa_packing_peanuts_l3651_365192


namespace NUMINAMATH_CALUDE_fraction_change_l3651_365132

/-- Given a fraction 3/4, if we increase the numerator by 12% and decrease the denominator by 2%,
    the resulting fraction is approximately 0.8571. -/
theorem fraction_change (ε : ℝ) (h_ε : ε > 0) :
  ∃ (new_fraction : ℝ),
    (3 * (1 + 0.12)) / (4 * (1 - 0.02)) = new_fraction ∧
    |new_fraction - 0.8571| < ε :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l3651_365132


namespace NUMINAMATH_CALUDE_f_at_one_l3651_365178

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_at_one_l3651_365178


namespace NUMINAMATH_CALUDE_beth_crayon_count_l3651_365149

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayon_count : total_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayon_count_l3651_365149


namespace NUMINAMATH_CALUDE_cheesecake_price_per_slice_l3651_365157

/-- Represents the price of a cheesecake slice -/
def price_per_slice (slices_per_pie : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (slices_per_pie * pies_sold)

/-- Proves that the price per slice of cheesecake is $7 -/
theorem cheesecake_price_per_slice :
  price_per_slice 6 7 294 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cheesecake_price_per_slice_l3651_365157


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3651_365118

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 25) = Real.sqrt 170 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3651_365118


namespace NUMINAMATH_CALUDE_coloring_book_coupons_l3651_365115

theorem coloring_book_coupons 
  (initial_stock : ℝ) 
  (books_sold : ℝ) 
  (coupons_per_book : ℝ) 
  (h1 : initial_stock = 40.0) 
  (h2 : books_sold = 20.0) 
  (h3 : coupons_per_book = 4.0) : 
  (initial_stock - books_sold) * coupons_per_book = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_coupons_l3651_365115


namespace NUMINAMATH_CALUDE_class_size_l3651_365113

/-- The number of chocolate bars Gerald brings --/
def gerald_bars : ℕ := 7

/-- The number of squares in each chocolate bar --/
def squares_per_bar : ℕ := 8

/-- The number of additional bars the teacher brings for each of Gerald's bars --/
def teacher_multiplier : ℕ := 2

/-- The number of squares each student gets --/
def squares_per_student : ℕ := 7

/-- The total number of chocolate bars --/
def total_bars : ℕ := gerald_bars + gerald_bars * teacher_multiplier

/-- The total number of chocolate squares --/
def total_squares : ℕ := total_bars * squares_per_bar

/-- The number of students in the class --/
def num_students : ℕ := total_squares / squares_per_student

theorem class_size : num_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3651_365113


namespace NUMINAMATH_CALUDE_rational_representation_l3651_365101

theorem rational_representation (q : ℚ) (hq : q > 0) :
  ∃ (a b c d : ℕ+), q = (a^2021 + b^2023) / (c^2022 + d^2024) := by
  sorry

end NUMINAMATH_CALUDE_rational_representation_l3651_365101


namespace NUMINAMATH_CALUDE_min_value_theorem_l3651_365127

theorem min_value_theorem (a b : ℝ) (h : a * b = 1) :
  4 * a^2 + 9 * b^2 ≥ 12 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3651_365127


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l3651_365109

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∃ x, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by
  sorry

-- Theorem 2
theorem range_of_m (m : ℝ) :
  (∀ x, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l3651_365109


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l3651_365126

theorem floor_abs_negative_real : ⌊|(-25.7 : ℝ)|⌋ = 25 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l3651_365126


namespace NUMINAMATH_CALUDE_earn_twelve_points_l3651_365136

/-- Calculates the points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_not_defeated : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_not_defeated) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 12 points --/
theorem earn_twelve_points :
  points_earned 6 2 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_earn_twelve_points_l3651_365136


namespace NUMINAMATH_CALUDE_article_cost_l3651_365196

/-- The cost of an article given specific selling price conditions -/
theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) 
  (price_difference : ℝ) (gain_difference_percent : ℝ) :
  selling_price_high = 350 →
  selling_price_low = 340 →
  price_difference = selling_price_high - selling_price_low →
  gain_difference_percent = 5 →
  price_difference = (gain_difference_percent / 100) * 200 →
  200 = price_difference / (gain_difference_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l3651_365196


namespace NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_sum_of_digits_l3651_365182

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Checks if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- Theorem: 99972 is the greatest five-digit number divisible by the sum of its digits -/
theorem greatest_five_digit_divisible_by_sum_of_digits :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ isDivisibleBySumOfDigits n → n ≤ 99972 :=
by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_divisible_by_sum_of_digits_l3651_365182


namespace NUMINAMATH_CALUDE_valid_count_is_48_l3651_365141

/-- Represents a three-digit number with the last two digits being the same -/
structure ThreeDigitNumber where
  first : Nat
  last : Nat
  first_is_digit : first ≤ 9
  last_is_digit : last ≤ 9

/-- Checks if a ThreeDigitNumber is valid according to the problem conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 11 * n.last) % 3 = 0 ∧
  n.first + 2 * n.last ≤ 18

/-- The count of valid ThreeDigitNumbers -/
def validCount : Nat :=
  (ThreeDigitNumber.mk 1 0 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 3 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 6 (by norm_num) (by norm_num) ::
   -- ... (other valid ThreeDigitNumbers)
   []).length

theorem valid_count_is_48 : validCount = 48 := by
  sorry

#eval validCount

end NUMINAMATH_CALUDE_valid_count_is_48_l3651_365141


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3651_365194

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal or both lines are vertical -/
def parallel (l1 l2 : Line) : Prop :=
  (l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b) ∨
  (l1.b = 0 ∧ l2.b = 0)

theorem parallel_lines_a_value (a : ℝ) :
  let l1 : Line := ⟨a, 2, a⟩
  let l2 : Line := ⟨3*a, a-1, 7⟩
  parallel l1 l2 → a = 0 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3651_365194


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3651_365175

theorem root_sum_theorem (p q r s : ℂ) : 
  p^4 - 15*p^3 + 35*p^2 - 27*p + 9 = 0 →
  q^4 - 15*q^3 + 35*q^2 - 27*q + 9 = 0 →
  r^4 - 15*r^3 + 35*r^2 - 27*r + 9 = 0 →
  s^4 - 15*s^3 + 35*s^2 - 27*s + 9 = 0 →
  p / (1/p + q*r) + q / (1/q + r*s) + r / (1/r + s*p) + s / (1/s + p*q) = 155/123 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3651_365175


namespace NUMINAMATH_CALUDE_factorization_of_4x_squared_plus_x_l3651_365114

theorem factorization_of_4x_squared_plus_x (x : ℝ) : 4 * x^2 + x = x * (4 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_squared_plus_x_l3651_365114


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3651_365181

theorem cost_price_calculation (marked_price : ℝ) (selling_price_percent : ℝ) (profit_percent : ℝ) :
  marked_price = 62.5 →
  selling_price_percent = 0.95 →
  profit_percent = 1.25 →
  ∃ (cost_price : ℝ), cost_price = 47.5 ∧ 
    selling_price_percent * marked_price = profit_percent * cost_price :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3651_365181


namespace NUMINAMATH_CALUDE_digit_replacement_theorem_l3651_365159

theorem digit_replacement_theorem : ∃ (x y z w : ℕ), 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 ∧
  42 * (10 * x + 8) = 2000 + 100 * y + 10 * z + w ∧
  (x + y + z + w) % 2 = 1 ∧
  2000 ≤ 42 * (10 * x + 8) ∧ 42 * (10 * x + 8) < 3000 :=
by sorry

end NUMINAMATH_CALUDE_digit_replacement_theorem_l3651_365159


namespace NUMINAMATH_CALUDE_inequality_condition_l3651_365125

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3651_365125


namespace NUMINAMATH_CALUDE_gcd_consecutive_pairs_l3651_365122

theorem gcd_consecutive_pairs (m n : ℕ) (h : m > n) :
  (∀ k : ℕ, k ∈ Finset.range (m - n) → Nat.gcd (n + k + 1) (m + k + 1) = 1) ↔ m = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_pairs_l3651_365122


namespace NUMINAMATH_CALUDE_black_tiles_201_implies_total_4624_l3651_365108

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- Calculates the number of black tiles on the floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  3 * floor.side_length - 3

/-- Calculates the total number of tiles on the floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If there are 201 black tiles, then the total number of tiles is 4624 -/
theorem black_tiles_201_implies_total_4624 :
  ∀ (floor : SquareFloor), black_tiles floor = 201 → total_tiles floor = 4624 :=
by
  sorry

end NUMINAMATH_CALUDE_black_tiles_201_implies_total_4624_l3651_365108


namespace NUMINAMATH_CALUDE_sequence_problem_l3651_365154

/-- The sequence function F that generates the nth term of the sequence --/
def F : ℕ → ℚ := sorry

/-- The sum of the first n natural numbers --/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that F(16) = 1/6 and F(4952) = 2/99 --/
theorem sequence_problem :
  F 16 = 1 / 6 ∧ F 4952 = 2 / 99 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l3651_365154


namespace NUMINAMATH_CALUDE_triangle_tangent_relation_l3651_365165

theorem triangle_tangent_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π / 2) ∧
  (0 < B) ∧ (B < π / 2) ∧
  (0 < C) ∧ (C < π / 2) ∧
  (A + B + C = π) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  (c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) ∧
  (Real.tan A * Real.tan B = Real.tan A * Real.tan C + Real.tan C * Real.tan B) →
  (a^2 + b^2) / c^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_relation_l3651_365165


namespace NUMINAMATH_CALUDE_inequality_solution_m_range_l3651_365138

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + m - 1

theorem inequality_solution (x : ℝ) :
  (m = -1 ∧ x ≥ 1) ∨
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) ↔
  f m x ≥ (m + 1) * x := by sorry

theorem m_range :
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f m x ≥ 0) → m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_m_range_l3651_365138


namespace NUMINAMATH_CALUDE_delta_phi_equation_l3651_365112

def δ (x : ℝ) : ℝ := 3 * x + 8

def φ (x : ℝ) : ℝ := 8 * x + 7

theorem delta_phi_equation (x : ℝ) : δ (φ x) = 7 ↔ x = -11/12 := by sorry

end NUMINAMATH_CALUDE_delta_phi_equation_l3651_365112


namespace NUMINAMATH_CALUDE_import_tax_threshold_l3651_365104

/-- The amount in excess of which the import tax was applied -/
def X : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that X is the correct amount in excess of which the import tax was applied -/
theorem import_tax_threshold (X total_value tax_rate tax_paid : ℝ) 
  (h1 : total_value = 2580)
  (h2 : tax_rate = 0.07)
  (h3 : tax_paid = 110.60) :
  X = 1000 ∧ tax_rate * (total_value - X) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_threshold_l3651_365104


namespace NUMINAMATH_CALUDE_problem_solution_l3651_365111

theorem problem_solution : 
  let x := ((12 ^ 5) * (6 ^ 4)) / ((3 ^ 2) * (36 ^ 2)) + (Real.sqrt 9 * Real.log 27)
  ∃ ε > 0, |x - 27657.887510597983| < ε := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3651_365111


namespace NUMINAMATH_CALUDE_floor_times_self_eq_90_l3651_365103

theorem floor_times_self_eq_90 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 90) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_90_l3651_365103


namespace NUMINAMATH_CALUDE_alpha_range_l3651_365151

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l3651_365151


namespace NUMINAMATH_CALUDE_ellipse_properties_l3651_365176

/-- An ellipse with center at the origin, foci on the x-axis, and eccentricity 1/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : c / a = 1 / 2
  h_rel : a^2 = b^2 + c^2

/-- A line passing through a point with a given angle -/
structure TangentLine where
  c : ℝ
  angle : ℝ
  h_angle : angle = π / 3

/-- The main theorem about the ellipse and its properties -/
theorem ellipse_properties (E : Ellipse) (L : TangentLine) :
  (E.a = 2 ∧ E.b = Real.sqrt 3 ∧ E.c = 1) ∧
  (∀ k m : ℝ, ∃ x y : ℝ,
    x^2 / 4 + y^2 / 3 = 1 ∧
    y = k * x + m ∧
    (x - 2)^2 + y^2 = 4 →
    k * (2 / 7) + m = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3651_365176


namespace NUMINAMATH_CALUDE_total_earnings_l3651_365145

/-- Calculate total earnings from selling candied apples and grapes -/
theorem total_earnings (num_apples : ℕ) (price_apple : ℚ) 
                       (num_grapes : ℕ) (price_grape : ℚ) : 
  num_apples = 15 → 
  price_apple = 2 → 
  num_grapes = 12 → 
  price_grape = (3/2) → 
  (num_apples : ℚ) * price_apple + (num_grapes : ℚ) * price_grape = 48 := by
sorry

end NUMINAMATH_CALUDE_total_earnings_l3651_365145


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l3651_365199

theorem smallest_sector_angle (n : ℕ) (a d : ℤ) : 
  n = 8 ∧ 
  (∀ i : ℕ, i < n → (a + i * d : ℤ) > 0) ∧
  (∀ i : ℕ, i < n → (a + i * d : ℤ).natAbs = a + i * d) ∧
  (n : ℤ) * (2 * a + (n - 1) * d) = 360 * 2 →
  a ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l3651_365199


namespace NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l3651_365179

theorem temperature_difference_product (P : ℝ) : 
  (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4) →
  (P = 17 ∨ P = 9) :=
by sorry

theorem product_of_possible_P_values : 
  (∃ P : ℝ, (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4)) →
  17 * 9 = 153 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_product_product_of_possible_P_values_l3651_365179


namespace NUMINAMATH_CALUDE_function_properties_imply_solution_set_l3651_365107

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def satisfies_negation_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def matches_linear_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (1/2) * x

def solution_set (k : ℤ) : ℝ := 4 * k - 1

theorem function_properties_imply_solution_set 
  (f : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : satisfies_negation_property f) 
  (h3 : matches_linear_on_interval f) :
  ∀ x, f x = -(1/2) ↔ ∃ k : ℤ, x = solution_set k :=
sorry

end NUMINAMATH_CALUDE_function_properties_imply_solution_set_l3651_365107


namespace NUMINAMATH_CALUDE_strawberries_in_buckets_l3651_365120

theorem strawberries_in_buckets (total_strawberries : ℕ) (num_buckets : ℕ) (removed_per_bucket : ℕ) :
  total_strawberries = 300 →
  num_buckets = 5 →
  removed_per_bucket = 20 →
  (total_strawberries / num_buckets) - removed_per_bucket = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberries_in_buckets_l3651_365120


namespace NUMINAMATH_CALUDE_square_sum_plus_product_squares_l3651_365160

theorem square_sum_plus_product_squares : (3 + 9)^2 + 3^2 * 9^2 = 873 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_plus_product_squares_l3651_365160


namespace NUMINAMATH_CALUDE_det_cofactor_matrix_cube_l3651_365161

/-- For a 4x4 matrix A, the determinant of its cofactor matrix B is equal to the cube of the determinant of A. -/
theorem det_cofactor_matrix_cube (A : Matrix (Fin 4) (Fin 4) ℝ) :
  let d := Matrix.det A
  let B := Matrix.adjugate A
  Matrix.det B = d^3 := by sorry

end NUMINAMATH_CALUDE_det_cofactor_matrix_cube_l3651_365161


namespace NUMINAMATH_CALUDE_unique_solution_l3651_365190

theorem unique_solution (x y z : ℝ) 
  (hx : x > 5) (hy : y > 5) (hz : z > 5)
  (h : ((x + 3)^2 / (y + z - 3)) + ((y + 5)^2 / (z + x - 5)) + ((z + 7)^2 / (x + y - 7)) = 45) :
  x = 15 ∧ y = 15 ∧ z = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l3651_365190


namespace NUMINAMATH_CALUDE_solution_set_equality_l3651_365183

theorem solution_set_equality : 
  {x : ℝ | (x - 3) * (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3651_365183


namespace NUMINAMATH_CALUDE_prism_faces_count_l3651_365137

/-- A prism is a polyhedron with two congruent polygonal bases and rectangular lateral faces. -/
structure Prism where
  /-- The number of sides in each base of the prism -/
  base_sides : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The sum of vertices and edges is 40 -/
  sum_condition : vertices + edges = 40
  /-- The number of vertices is twice the number of base sides -/
  vertices_def : vertices = 2 * base_sides
  /-- The number of edges is thrice the number of base sides -/
  edges_def : edges = 3 * base_sides
  /-- The number of faces is 2 more than the number of base sides -/
  faces_def : faces = base_sides + 2

/-- Theorem: A prism with 40 as the sum of its edges and vertices has 10 faces -/
theorem prism_faces_count (p : Prism) : p.faces = 10 := by
  sorry


end NUMINAMATH_CALUDE_prism_faces_count_l3651_365137


namespace NUMINAMATH_CALUDE_smallest_an_l3651_365134

theorem smallest_an (n : ℕ+) (x : ℝ) :
  (((x^(2^(n.val+1)) + 1) / 2) ^ (1 / (2^n.val))) ≤ 2^(n.val-1) * (x-1)^2 + x :=
sorry

end NUMINAMATH_CALUDE_smallest_an_l3651_365134


namespace NUMINAMATH_CALUDE_area_increase_l3651_365100

/-- A rectangle with perimeter 160 meters -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 80

/-- The change in area when increasing both sides by 10 meters -/
def area_change (rect : Rectangle) : ℝ :=
  (rect.length + 10) * (rect.width + 10) - rect.length * rect.width

theorem area_increase (rect : Rectangle) : area_change rect = 900 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_l3651_365100


namespace NUMINAMATH_CALUDE_equation_solution_l3651_365180

theorem equation_solution :
  let s : ℚ := 20
  let r : ℚ := 270 / 7
  2 * (r - 45) / 3 = (3 * s - 2 * r) / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3651_365180


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l3651_365152

theorem prop_a_necessary_not_sufficient :
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) ∧
  (∀ a : ℝ, a^2 < 4 → a < 2) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l3651_365152


namespace NUMINAMATH_CALUDE_basketball_expected_score_l3651_365168

def expected_score (p_in : ℝ) (p_out : ℝ) (n_in : ℕ) (n_out : ℕ) (points_in : ℕ) (points_out : ℕ) : ℝ :=
  (p_in * n_in * points_in) + (p_out * n_out * points_out)

theorem basketball_expected_score :
  expected_score 0.7 0.4 10 5 2 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_expected_score_l3651_365168


namespace NUMINAMATH_CALUDE_train_tickets_theorem_l3651_365119

/-- Calculates the number of different tickets needed for a train route -/
def number_of_tickets (intermediate_stops : ℕ) : ℕ :=
  intermediate_stops * (intermediate_stops + 3) + 2

/-- Theorem stating that a train route with 5 intermediate stops requires 42 different tickets -/
theorem train_tickets_theorem :
  number_of_tickets 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_tickets_theorem_l3651_365119


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l3651_365191

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h₁ : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h₂ : 2 * π * r₁ + 2 * π * r₂ = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l3651_365191


namespace NUMINAMATH_CALUDE_inequality_proof_l3651_365128

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3651_365128


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3651_365121

-- Define the sample space
def sampleSpace : ℕ := 10 -- (5 choose 2)

-- Define the events
def exactlyOneMale (outcome : ℕ) : Prop := sorry
def exactlyTwoFemales (outcome : ℕ) : Prop := sorry

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

-- Define complementary events
def complementary (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, e1 outcome ↔ ¬(e2 outcome)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneMale exactlyTwoFemales ∧
  ¬(complementary exactlyOneMale exactlyTwoFemales) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l3651_365121


namespace NUMINAMATH_CALUDE_triangle_properties_l3651_365123

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  2 * Real.cos (t.A + t.B) = -1

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = Real.pi / 3 ∧
  t.c = Real.sqrt 6 ∧
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3651_365123


namespace NUMINAMATH_CALUDE_money_difference_l3651_365164

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 7

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem money_difference : difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3651_365164


namespace NUMINAMATH_CALUDE_julia_tag_game_l3651_365130

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) : 
  monday_kids = 22 → 
  monday_kids = tuesday_kids + 8 → 
  tuesday_kids = 14 := by
sorry

end NUMINAMATH_CALUDE_julia_tag_game_l3651_365130


namespace NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l3651_365172

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line x - y + 1 = 0
def line_1 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the x-axis
def x_axis (y : ℝ) : Prop := y = 0

-- Define the line x + y + 3 = 0
def line_2 (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle x^2 + (y - 3)^2 = 4
def circle_C2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -1 ∨ 4*x - 3*y + 4 = 0

-- Theorem 1
theorem circle_C_equation : 
  ∀ x y : ℝ, 
  (∃ x₀, line_1 x₀ 0 ∧ x_axis 0) → 
  (∀ x₁ y₁, line_2 x₁ y₁ → ∃ t, circle_C (x₁ + t) (y₁ + t) ∧ ¬(∃ s ≠ t, circle_C (x₁ + s) (y₁ + s))) →
  circle_C x y :=
sorry

-- Theorem 2
theorem line_l_equation :
  ∀ x y : ℝ,
  circle_C2 x y →
  (∃ x₀ y₀, x₀ = -1 ∧ y₀ = 0 ∧ line_l x₀ y₀) →
  (∃ p q : ℝ × ℝ, circle_C2 p.1 p.2 ∧ circle_C2 q.1 q.2 ∧ line_l p.1 p.2 ∧ line_l q.1 q.2 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 12) →
  line_l x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l3651_365172


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l3651_365129

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ) : 
  total = 120 →
  red_fraction = 2/3 →
  yellow_fraction = 1 - red_fraction →
  (yellow_fraction * total : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l3651_365129


namespace NUMINAMATH_CALUDE_three_digit_number_is_142_l3651_365147

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a repeating decimal of the form 0.xyxy̅xy to a fraction -/
def repeating_decimal_xy (x y : Digit) : ℚ :=
  (10 * x.val + y.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.xyzxyz̅xyz to a fraction -/
def repeating_decimal_xyz (x y z : Digit) : ℚ :=
  (100 * x.val + 10 * y.val + z.val : ℚ) / 999

/-- The main theorem stating that the three-digit number xyz is 142 -/
theorem three_digit_number_is_142 :
  ∃ (x y z : Digit),
    repeating_decimal_xy x y + repeating_decimal_xyz x y z = 39 / 41 ∧
    x.val = 1 ∧ y.val = 4 ∧ z.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_is_142_l3651_365147


namespace NUMINAMATH_CALUDE_shape_count_theorem_l3651_365148

/-- Represents the count of shapes in a box -/
structure ShapeCount where
  triangles : ℕ
  squares : ℕ
  circles : ℕ

/-- Checks if a ShapeCount satisfies the given conditions -/
def isValidShapeCount (sc : ShapeCount) : Prop :=
  sc.triangles + sc.squares + sc.circles = 24 ∧
  sc.triangles = 7 * sc.squares

/-- The set of all possible valid shape counts -/
def validShapeCounts : Set ShapeCount :=
  { sc | isValidShapeCount sc }

/-- The theorem stating the only possible combinations -/
theorem shape_count_theorem :
  validShapeCounts = {
    ⟨0, 0, 24⟩,
    ⟨7, 1, 16⟩,
    ⟨14, 2, 8⟩,
    ⟨21, 3, 0⟩
  } := by sorry

end NUMINAMATH_CALUDE_shape_count_theorem_l3651_365148


namespace NUMINAMATH_CALUDE_intersection_line_parabola_l3651_365105

/-- The line y = kx - 2 intersects the parabola y² = 8x at two points A and B,
    and the x-coordinate of the midpoint of AB is 2. Then k = 2. -/
theorem intersection_line_parabola (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    (∀ x y, (x, y) = A ∨ (x, y) = B → y = k * x - 2 ∧ y^2 = 8 * x) ∧
    (A.1 + B.1) / 2 = 2) →
  k = 2 := by
sorry


end NUMINAMATH_CALUDE_intersection_line_parabola_l3651_365105


namespace NUMINAMATH_CALUDE_existence_of_abc_l3651_365177

theorem existence_of_abc (n k : ℕ) (h1 : n > 20) (h2 : k > 1) (h3 : k^2 ∣ n) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l3651_365177


namespace NUMINAMATH_CALUDE_problem_solution_l3651_365173

theorem problem_solution : 
  (Real.sqrt 6 + Real.sqrt 8 * Real.sqrt 12 = 5 * Real.sqrt 6) ∧ 
  (Real.sqrt 4 - Real.sqrt 2 / (Real.sqrt 2 + 1) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3651_365173


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l3651_365186

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)
  (is_right : side1 = 5 ∧ side2 = 12 ∧ hypotenuse = 13)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (is_inscribed : True)  -- We assume the square is properly inscribed

/-- The side length of the inscribed square is 780/169 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 780 / 169 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l3651_365186


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3651_365163

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 13) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 13 ∧ a + b = 196 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 13 → c + d ≥ 196 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3651_365163


namespace NUMINAMATH_CALUDE_bus_fraction_proof_l3651_365187

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_proof :
  let distance_by_foot : ℝ := (1/3) * total_distance
  let distance_by_car : ℝ := 2
  let distance_by_bus : ℝ := total_distance - distance_by_foot - distance_by_car
  distance_by_bus / total_distance = 3/5 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_proof_l3651_365187


namespace NUMINAMATH_CALUDE_base_subtraction_l3651_365166

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_subtraction : 
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_7_number := [5, 6, 1]  -- 165 in base 7 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_7_number 7) = 169 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l3651_365166


namespace NUMINAMATH_CALUDE_montoya_family_food_budget_l3651_365169

theorem montoya_family_food_budget (grocery_fraction eating_out_fraction : ℝ) 
  (h1 : grocery_fraction = 0.6)
  (h2 : eating_out_fraction = 0.2) :
  grocery_fraction + eating_out_fraction = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_montoya_family_food_budget_l3651_365169


namespace NUMINAMATH_CALUDE_boys_age_problem_l3651_365185

theorem boys_age_problem (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 ∧ age2 = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_age_problem_l3651_365185
