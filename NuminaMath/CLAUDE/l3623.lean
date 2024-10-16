import Mathlib

namespace NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l3623_362324

theorem x_gt_y_necessary_not_sufficient (x y : ℝ) (hx : x > 0) :
  (∀ y, x > |y| → x > y) ∧ ¬(∀ y, x > y → x > |y|) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_y_necessary_not_sufficient_l3623_362324


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3623_362336

theorem right_triangle_hypotenuse (x y z : ℝ) : 
  x > 0 → 
  y > 0 → 
  z > 0 → 
  y = 3 * x - 3 → 
  (1 / 2) * x * y = 72 → 
  x^2 + y^2 = z^2 → 
  z = Real.sqrt 505 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3623_362336


namespace NUMINAMATH_CALUDE_tan_75_deg_l3623_362345

/-- Proves that tan 75° = 2 + √3 given tan 60° and tan 15° -/
theorem tan_75_deg (tan_60_deg : Real.tan (60 * π / 180) = Real.sqrt 3)
                   (tan_15_deg : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) :
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_deg_l3623_362345


namespace NUMINAMATH_CALUDE_initial_girls_count_l3623_362388

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 12) = b) →
  (4 * (b - 36) = g - 12) →
  g = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3623_362388


namespace NUMINAMATH_CALUDE_road_system_exists_road_system_impossible_l3623_362371

/-- A graph representing the road system in the kingdom --/
structure RoadSystem where
  cities : Finset ℕ
  roads : cities → cities → Prop

/-- The distance between two cities in the road system --/
def distance (G : RoadSystem) (a b : G.cities) : ℕ :=
  sorry

/-- The degree (number of outgoing roads) of a city in the road system --/
def degree (G : RoadSystem) (a : G.cities) : ℕ :=
  sorry

/-- Theorem stating the existence of a road system satisfying the king's requirements --/
theorem road_system_exists :
  ∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 5) :=
  sorry

/-- Theorem stating the impossibility of a road system with reduced maximum degree --/
theorem road_system_impossible :
  ¬∃ (G : RoadSystem),
    G.cities.card = 16 ∧
    (∀ a b : G.cities, a ≠ b → distance G a b ≤ 2) ∧
    (∀ a : G.cities, degree G a ≤ 4) :=
  sorry

end NUMINAMATH_CALUDE_road_system_exists_road_system_impossible_l3623_362371


namespace NUMINAMATH_CALUDE_max_regions_1002_1000_l3623_362344

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of new regions added by each line through A after the first -/
def new_regions_per_line_A (lines_through_B : ℕ) : ℕ := lines_through_B + 2

/-- The maximum number of regions formed by m lines through A and n lines through B -/
def max_regions_two_points (m n : ℕ) : ℕ :=
  max_regions n + (new_regions_per_line_A n) + (m - 1) * (new_regions_per_line_A n)

theorem max_regions_1002_1000 :
  max_regions_two_points 1002 1000 = 1504503 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_1002_1000_l3623_362344


namespace NUMINAMATH_CALUDE_ratio_of_15th_terms_l3623_362329

/-- Two arithmetic sequences with sums S_n and T_n for the first n terms -/
def arithmetic_sequences (S T : ℕ → ℚ) : Prop :=
  ∃ (a d b e : ℚ), ∀ n : ℕ,
    S n = n / 2 * (2 * a + (n - 1) * d) ∧
    T n = n / 2 * (2 * b + (n - 1) * e)

/-- The ratio condition for the sums -/
def ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n / T n = (9 * n + 5) / (6 * n + 31)

/-- The 15th term of an arithmetic sequence -/
def term_15 (a d : ℚ) : ℚ := a + 14 * d

/-- Main theorem -/
theorem ratio_of_15th_terms 
  (S T : ℕ → ℚ) 
  (h1 : arithmetic_sequences S T) 
  (h2 : ratio_condition S T) : 
  ∃ (a d b e : ℚ), 
    term_15 a d / term_15 b e = 92 / 71 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_15th_terms_l3623_362329


namespace NUMINAMATH_CALUDE_average_transformation_l3623_362359

theorem average_transformation (t b c : ℝ) :
  (t + b + c + 14 + 15) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_transformation_l3623_362359


namespace NUMINAMATH_CALUDE_rals_age_l3623_362375

/-- Given that Ral's age is twice Suri's age and Suri's age plus 3 years equals 16 years,
    prove that Ral's current age is 26 years. -/
theorem rals_age (suri_age : ℕ) (ral_age : ℕ) : 
  ral_age = 2 * suri_age → 
  suri_age + 3 = 16 → 
  ral_age = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_rals_age_l3623_362375


namespace NUMINAMATH_CALUDE_electric_blankets_sold_l3623_362380

/-- Represents the number of electric blankets sold -/
def electric_blankets : ℕ := sorry

/-- Represents the number of hot-water bottles sold -/
def hot_water_bottles : ℕ := sorry

/-- Represents the number of thermometers sold -/
def thermometers : ℕ := sorry

/-- The price of a thermometer in dollars -/
def thermometer_price : ℕ := 2

/-- The price of a hot-water bottle in dollars -/
def hot_water_bottle_price : ℕ := 6

/-- The price of an electric blanket in dollars -/
def electric_blanket_price : ℕ := 10

/-- The total sales for all items in dollars -/
def total_sales : ℕ := 1800

theorem electric_blankets_sold :
  (thermometer_price * thermometers + 
   hot_water_bottle_price * hot_water_bottles + 
   electric_blanket_price * electric_blankets = total_sales) ∧
  (thermometers = 7 * hot_water_bottles) ∧
  (hot_water_bottles = 2 * electric_blankets) →
  electric_blankets = 36 := by sorry

end NUMINAMATH_CALUDE_electric_blankets_sold_l3623_362380


namespace NUMINAMATH_CALUDE_area_of_JKLMNO_l3623_362304

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (J K L M N O : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Calculate the area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The given polygon JKLMNO -/
def polygon : Hexagon := sorry

/-- The intersection point P -/
def P : Point := sorry

/-- Theorem: The area of polygon JKLMNO is 62 square units -/
theorem area_of_JKLMNO : 
  let JK : ℝ := 8
  let KL : ℝ := 10
  let OP : ℝ := 6
  let PM : ℝ := 3
  let area_JKLMNP := rectangleArea JK KL
  let area_PMNO := rectangleArea PM OP
  area_JKLMNP - area_PMNO = 62 := by sorry

end NUMINAMATH_CALUDE_area_of_JKLMNO_l3623_362304


namespace NUMINAMATH_CALUDE_equality_and_inequality_proof_l3623_362325

theorem equality_and_inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (3 : ℝ)^x = (4 : ℝ)^y ∧ (4 : ℝ)^y = (6 : ℝ)^z) : 
  (1 / z - 1 / x = 1 / (2 * y)) ∧ (3 * x < 4 * y ∧ 4 * y < 6 * z) := by
sorry

end NUMINAMATH_CALUDE_equality_and_inequality_proof_l3623_362325


namespace NUMINAMATH_CALUDE_polynomial_sign_intervals_l3623_362334

theorem polynomial_sign_intervals (x : ℝ) :
  x > 0 → ((x - 1) * (x - 2) * (x - 3) < 0 ↔ (x > 0 ∧ x < 1) ∨ (x > 2 ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sign_intervals_l3623_362334


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3623_362387

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  a 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3623_362387


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3623_362305

theorem factorization_a_squared_minus_ab (a b : ℝ) : a^2 - a*b = a*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_ab_l3623_362305


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3623_362349

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (5*b - 6) = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3623_362349


namespace NUMINAMATH_CALUDE_levis_brother_additional_scores_l3623_362376

/-- Proves that Levi's brother scored 3 more times given the initial conditions and Levi's goal -/
theorem levis_brother_additional_scores :
  ∀ (levi_initial : ℕ) (brother_initial : ℕ) (levi_additional : ℕ) (goal_difference : ℕ),
    levi_initial = 8 →
    brother_initial = 12 →
    levi_additional = 12 →
    goal_difference = 5 →
    ∃ (brother_additional : ℕ),
      levi_initial + levi_additional = brother_initial + brother_additional + goal_difference ∧
      brother_additional = 3 :=
by sorry

end NUMINAMATH_CALUDE_levis_brother_additional_scores_l3623_362376


namespace NUMINAMATH_CALUDE_distributions_without_zhoubi_l3623_362319

/-- Represents the number of books -/
def num_books : ℕ := 4

/-- Represents the number of students -/
def num_students : ℕ := 3

/-- Represents the total number of distribution methods -/
def total_distributions : ℕ := 36

/-- Represents the number of distribution methods where student A receives "Zhoubi Suanjing" -/
def distributions_with_zhoubi : ℕ := 12

/-- Theorem stating the number of distribution methods where A does not receive "Zhoubi Suanjing" -/
theorem distributions_without_zhoubi :
  total_distributions - distributions_with_zhoubi = 24 :=
by sorry

end NUMINAMATH_CALUDE_distributions_without_zhoubi_l3623_362319


namespace NUMINAMATH_CALUDE_difference_in_circumferences_l3623_362318

/-- The difference in circumferences of two concentric circles -/
theorem difference_in_circumferences 
  (inner_diameter : ℝ) 
  (track_width : ℝ) 
  (h1 : inner_diameter = 50) 
  (h2 : track_width = 15) : 
  (inner_diameter + 2 * track_width) * π - inner_diameter * π = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_difference_in_circumferences_l3623_362318


namespace NUMINAMATH_CALUDE_root_existence_implies_n_range_l3623_362397

-- Define the function f
def f (m n x : ℝ) : ℝ := m * x^2 - (5 * m + n) * x + n

-- State the theorem
theorem root_existence_implies_n_range :
  (∀ m ∈ Set.Ioo (-2 : ℝ) (-1 : ℝ),
    ∃ x ∈ Set.Ioo (3 : ℝ) (5 : ℝ), f m n x = 0) →
  n ∈ Set.Ioo (0 : ℝ) (3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_implies_n_range_l3623_362397


namespace NUMINAMATH_CALUDE_max_min_sum_l3623_362390

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 2

-- Define the maximum and minimum values
def M (a b : ℝ) : ℝ := sorry
def m (a b : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_min_sum (a b : ℝ) (h : a ≠ 0) : M a b + m a b = 4 := by sorry

end NUMINAMATH_CALUDE_max_min_sum_l3623_362390


namespace NUMINAMATH_CALUDE_exists_a_with_median_4_l3623_362367

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (λ x => x ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ x => x ≥ m)).card ≥ s.card

theorem exists_a_with_median_4 : 
  ∃ a : ℝ, is_median {a, 2, 4, 0, 5} 4 := by
sorry

end NUMINAMATH_CALUDE_exists_a_with_median_4_l3623_362367


namespace NUMINAMATH_CALUDE_ellipse_properties_l3623_362355

/-- The ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 5 / 9
  h_minor : b = 2

/-- The condition for the line x = m -/
def line_condition (C : Ellipse) (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 →
  x ≠ -C.a ∧ x ≠ C.a →
  (x - C.a) * (5 / 9 * m - 13 / 3) = 0

theorem ellipse_properties (C : Ellipse) :
  C.a^2 = 9 ∧ C.b^2 = 4 ∧ line_condition C (39 / 5) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3623_362355


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3623_362358

def proposition (x : ℕ) : Prop := (1/2:ℝ)^x ≤ 1/2

theorem negation_of_proposition :
  (¬ ∀ (x : ℕ), x > 0 → proposition x) ↔ (∃ (x : ℕ), x > 0 ∧ (1/2:ℝ)^x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3623_362358


namespace NUMINAMATH_CALUDE_average_movers_to_texas_l3623_362309

/-- The number of people moving to Texas over 5 days -/
def total_people : ℕ := 3500

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem average_movers_to_texas :
  round_to_nearest average_per_hour = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_movers_to_texas_l3623_362309


namespace NUMINAMATH_CALUDE_quad_func_order_l3623_362383

/-- A quadratic function with the given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  f : ℝ → ℝ
  h_f_def : ∀ x, f x = a * x^2 + b * x + c
  h_f_sym : ∀ x, f (x + 2) = f (2 - x)

/-- The main theorem stating the order of function values -/
theorem quad_func_order (qf : QuadraticFunction) :
  qf.f (-1992) < qf.f 1992 ∧ qf.f 1992 < qf.f 0 := by
  sorry

end NUMINAMATH_CALUDE_quad_func_order_l3623_362383


namespace NUMINAMATH_CALUDE_angle_expression_simplification_l3623_362301

theorem angle_expression_simplification (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) (h3 : Real.cos α = -Real.sqrt 5 / 5) :
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.tan (-α - π) * Real.sin (-π - α)) = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_angle_expression_simplification_l3623_362301


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l3623_362378

/-- The number of ways to distribute n identical balls into k distinct boxes, leaving no box empty -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- Theorem stating that there are 20 ways to distribute 7 identical balls into 4 distinct boxes with no empty box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l3623_362378


namespace NUMINAMATH_CALUDE_conference_room_chairs_l3623_362354

/-- The number of chairs in the conference room -/
def num_chairs : ℕ := 40

/-- The capacity of each chair -/
def chair_capacity : ℕ := 2

/-- The fraction of unoccupied chairs -/
def unoccupied_fraction : ℚ := 2/5

/-- The number of board members who attended the meeting -/
def attendees : ℕ := 48

theorem conference_room_chairs :
  (num_chairs : ℚ) * chair_capacity * (1 - unoccupied_fraction) = attendees ∧
  num_chairs * chair_capacity = num_chairs * 2 :=
sorry

end NUMINAMATH_CALUDE_conference_room_chairs_l3623_362354


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l3623_362311

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l3623_362311


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l3623_362303

theorem inequality_of_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z)) + Real.sqrt (y / (z + x)) + Real.sqrt (z / (x + y)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l3623_362303


namespace NUMINAMATH_CALUDE_f_properties_l3623_362370

-- Define the function f(x) = x^3 + ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

-- State the theorem
theorem f_properties (a : ℝ) (h : a > 0) :
  -- f(x) has exactly two critical points
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂)) ∧
  -- The point (-a/3, f(-a/3)) is the center of symmetry
  (∀ x : ℝ, f a (-a/3 + x) = f a (-a/3 - x)) ∧
  -- There exists a point where y = x is tangent to y = f(x)
  (∃ x₀ : ℝ, deriv (f a) x₀ = 1 ∧ f a x₀ = x₀) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l3623_362370


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l3623_362320

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l3623_362320


namespace NUMINAMATH_CALUDE_optimal_investment_l3623_362332

/-- Represents the profit function for the company's investments -/
def profit_function (x : ℝ) : ℝ :=
  let t := 3 - x
  (-x^3 + x^2 + 3*x) + (-t^2 + 5*t) - 3

/-- Theorem stating the optimal investment allocation and maximum profit -/
theorem optimal_investment :
  ∃ (x : ℝ), 
    0 ≤ x ∧ 
    x ≤ 3 ∧ 
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ 3 → profit_function x ≥ profit_function y ∧
    x = 2 ∧
    profit_function x = 25/3 := by
  sorry

#check optimal_investment

end NUMINAMATH_CALUDE_optimal_investment_l3623_362332


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3623_362328

/-- 
Given a rectangular solid with edge lengths a, b, and c,
if the total surface area is 22 and the total edge length is 24,
then the length of any interior diagonal is √14.
-/
theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 22)
  (h2 : 4 * (a + b + c) = 24) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3623_362328


namespace NUMINAMATH_CALUDE_first_player_win_prob_l3623_362337

/-- The probability of the first player getting heads -/
def p1 : ℚ := 1/3

/-- The probability of the second player getting heads -/
def p2 : ℚ := 2/5

/-- The game where two players flip coins alternately until one gets heads -/
def coin_flip_game (p1 p2 : ℚ) : Prop :=
  p1 > 0 ∧ p1 < 1 ∧ p2 > 0 ∧ p2 < 1

/-- The probability of the first player winning the game -/
noncomputable def win_prob (p1 p2 : ℚ) : ℚ :=
  p1 / (1 - (1 - p1) * (1 - p2))

/-- Theorem stating that the probability of the first player winning is 5/9 -/
theorem first_player_win_prob :
  coin_flip_game p1 p2 → win_prob p1 p2 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_first_player_win_prob_l3623_362337


namespace NUMINAMATH_CALUDE_eighth_term_value_l3623_362382

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem eighth_term_value (a : ℕ → ℚ) 
  (h1 : a 2 = 3)
  (h2 : a 5 = 1)
  (h3 : is_arithmetic_sequence (fun n ↦ 1 / (a n + 1))) :
  a 8 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l3623_362382


namespace NUMINAMATH_CALUDE_parities_of_E_10_11_12_l3623_362379

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => 2 * E (n + 2) + E n

theorem parities_of_E_10_11_12 :
  Even (E 10) ∧ Odd (E 11) ∧ Odd (E 12) := by
  sorry

end NUMINAMATH_CALUDE_parities_of_E_10_11_12_l3623_362379


namespace NUMINAMATH_CALUDE_log_expression_equals_21_l3623_362326

theorem log_expression_equals_21 :
  2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 - Real.log (Real.log (3^10) / Real.log 3) = 21 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_21_l3623_362326


namespace NUMINAMATH_CALUDE_forty_percent_of_jacqueline_candy_bars_l3623_362399

def fred_candy_bars : ℕ := 12
def uncle_bob_extra_candy_bars : ℕ := 6
def jacqueline_multiplier : ℕ := 10

def uncle_bob_candy_bars : ℕ := fred_candy_bars + uncle_bob_extra_candy_bars
def total_fred_and_bob : ℕ := fred_candy_bars + uncle_bob_candy_bars
def jacqueline_candy_bars : ℕ := jacqueline_multiplier * total_fred_and_bob

theorem forty_percent_of_jacqueline_candy_bars : 
  (40 : ℕ) * jacqueline_candy_bars / 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_jacqueline_candy_bars_l3623_362399


namespace NUMINAMATH_CALUDE_allen_pizza_change_l3623_362364

def pizza_order (num_boxes : ℕ) (price_per_box : ℚ) (tip_fraction : ℚ) (payment : ℚ) : ℚ :=
  let total_cost := num_boxes * price_per_box
  let tip := total_cost * tip_fraction
  let total_spent := total_cost + tip
  payment - total_spent

theorem allen_pizza_change : 
  pizza_order 5 7 (1/7) 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_allen_pizza_change_l3623_362364


namespace NUMINAMATH_CALUDE_sample_size_eq_selected_size_world_book_day_survey_sample_size_l3623_362368

/-- Represents a survey conducted on a population -/
structure Survey where
  population_size : ℕ
  selected_size : ℕ

/-- The sample size of a survey is equal to the number of selected individuals -/
theorem sample_size_eq_selected_size (s : Survey) : s.selected_size = 50 → s.selected_size = s.selected_size := by
  sorry

/-- Given the specific survey conditions, prove that the sample size is 50 -/
theorem world_book_day_survey_sample_size : 
  ∃ (s : Survey), s.population_size = 350 ∧ s.selected_size = 50 ∧ s.selected_size = 50 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_eq_selected_size_world_book_day_survey_sample_size_l3623_362368


namespace NUMINAMATH_CALUDE_minimum_points_to_win_l3623_362351

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult
| Other  : RaceResult

/-- Converts a race result to points -/
def pointsForResult (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 4
  | RaceResult.Second => 2
  | RaceResult.Third  => 1
  | RaceResult.Other  => 0

/-- Calculates total points for a series of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map pointsForResult |>.sum

/-- Represents all possible combinations of race results for four races -/
def allPossibleResults : List (List RaceResult) :=
  sorry

theorem minimum_points_to_win (results : List RaceResult) :
  (results.length = 4) →
  (totalPoints results ≥ 15) →
  (∀ other : List RaceResult, other.length = 4 → totalPoints other < totalPoints results) :=
sorry

end NUMINAMATH_CALUDE_minimum_points_to_win_l3623_362351


namespace NUMINAMATH_CALUDE_museum_artifacts_l3623_362385

theorem museum_artifacts (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wings : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_multiplier : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wings = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting_wings + small_painting_wings * paintings_per_small_wing
  let total_artifacts := total_paintings * artifact_multiplier
  let artifact_wings := total_wings - painting_wings
  ∀ wing, wing ≤ artifact_wings → 
    (total_artifacts / artifact_wings : Nat) = 20 := by
  sorry

#check museum_artifacts

end NUMINAMATH_CALUDE_museum_artifacts_l3623_362385


namespace NUMINAMATH_CALUDE_factorization_problem_l3623_362347

theorem factorization_problem (a m n b : ℝ) : 
  (∀ x, x^2 + a*x + m = (x + 2) * (x + 4)) →
  (∀ x, x^2 + n*x + b = (x + 1) * (x + 9)) →
  (∀ x, x^2 + a*x + b = (x + 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problem_l3623_362347


namespace NUMINAMATH_CALUDE_card_purchase_cost_l3623_362341

/-- Calculates the total cost of cards purchased from two boxes, including sales tax. -/
def total_cost (price1 : ℚ) (price2 : ℚ) (count1 : ℕ) (count2 : ℕ) (tax_rate : ℚ) : ℚ :=
  let subtotal := price1 * count1 + price2 * count2
  subtotal * (1 + tax_rate)

/-- Proves that the total cost of 8 cards from the first box and 12 cards from the second box, including 7% sales tax, is $33.17. -/
theorem card_purchase_cost : 
  total_cost (25/20) (35/20) 8 12 (7/100) = 3317/100 := by
  sorry

#eval total_cost (25/20) (35/20) 8 12 (7/100)

end NUMINAMATH_CALUDE_card_purchase_cost_l3623_362341


namespace NUMINAMATH_CALUDE_walkway_and_border_area_is_912_l3623_362377

/-- Represents the dimensions and layout of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  border_width : ℕ

/-- Calculates the total area of walkways and decorative border in the garden -/
def walkway_and_border_area (g : Garden) : ℕ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width + 2 * g.border_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width + 2 * g.border_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - beds_area

/-- Theorem stating that the walkway and border area for the given garden specifications is 912 square feet -/
theorem walkway_and_border_area_is_912 :
  walkway_and_border_area ⟨4, 3, 8, 3, 2, 4⟩ = 912 := by
  sorry

end NUMINAMATH_CALUDE_walkway_and_border_area_is_912_l3623_362377


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3623_362330

/-- Given an infinite geometric series with first term a and common ratio r,
    if the sum of the series is 20 and the sum of cubes of its terms is 80,
    then the first term a is approximately 3.42. -/
theorem geometric_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20)
  (h2 : a^3 / (1 - r^3) = 80)
  (h3 : 0 ≤ r ∧ r < 1) : 
  ‖a - 3.42‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3623_362330


namespace NUMINAMATH_CALUDE_max_glass_height_l3623_362357

/-- The maximum height of a truncated cone-shaped glass that can roll around a circular table without reaching the edge -/
theorem max_glass_height (table_diameter : Real) (glass_bottom_diameter : Real) (glass_top_diameter : Real)
  (h_table : table_diameter = 160)
  (h_glass_bottom : glass_bottom_diameter = 5)
  (h_glass_top : glass_top_diameter = 6.5) :
  ∃ (max_height : Real), 
    (∀ (h : Real), h > 0 ∧ h < max_height → 
      ∃ (x y : Real), x^2 + y^2 < (table_diameter/2)^2 ∧ 
        ((h * glass_bottom_diameter/2) / (glass_top_diameter/2 - glass_bottom_diameter/2))^2 + h^2 = 
        ((y - x) * (glass_top_diameter/2 - glass_bottom_diameter/2) / h)^2) ∧
    max_height < (3/13) * Real.sqrt 6389.4375 := by
  sorry

end NUMINAMATH_CALUDE_max_glass_height_l3623_362357


namespace NUMINAMATH_CALUDE_cos_sin_75_product_equality_l3623_362348

theorem cos_sin_75_product_equality : 
  (Real.cos (75 * π / 180) + Real.sin (75 * π / 180)) * 
  (Real.cos (75 * π / 180) - Real.sin (75 * π / 180)) = 
  - (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sin_75_product_equality_l3623_362348


namespace NUMINAMATH_CALUDE_stratified_sampling_l3623_362362

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The stratified sampling theorem -/
theorem stratified_sampling
  (pop : Population)
  (sample : Sample)
  (h1 : pop.elderly = 27)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (h4 : sample.elderly = 6)
  (h5 : sample.middleAged / pop.middleAged = sample.elderly / pop.elderly)
  (h6 : sample.young / pop.young = sample.elderly / pop.elderly) :
  sample.elderly + sample.middleAged + sample.young = 36 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3623_362362


namespace NUMINAMATH_CALUDE_radical_simplification_l3623_362384

theorem radical_simplification (x : ℝ) (h : 4 < x ∧ x < 7) : 
  (((x - 4) ^ 4) ^ (1/4)) + (((x - 7) ^ 4) ^ (1/4)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3623_362384


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l3623_362361

/-- A sequence of integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℤ) :=
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ i, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) + (a (i+5)) + (a (i+6)) < 0)

/-- The maximum length of a valid sequence is 10 -/
theorem max_valid_sequence_length :
  (∃ (a : ℕ → ℤ) (n : ℕ), n = 10 ∧ ValidSequence (λ i => if i < n then a i else 0)) ∧
  (∀ (a : ℕ → ℤ) (n : ℕ), n > 10 → ¬ValidSequence (λ i => if i < n then a i else 0)) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l3623_362361


namespace NUMINAMATH_CALUDE_cubic_odd_extremum_sum_l3623_362333

/-- A cubic function f(x) = ax³ + bx² + cx -/
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f has an extremum at x=1 -/
def has_extremum_at_one (f : ℝ → ℝ) : Prop := 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

theorem cubic_odd_extremum_sum (a b c : ℝ) : 
  is_odd_function (f a b c) → has_extremum_at_one (f a b c) → 3*a + b + c = 0 := by
  sorry


end NUMINAMATH_CALUDE_cubic_odd_extremum_sum_l3623_362333


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3623_362372

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l3623_362372


namespace NUMINAMATH_CALUDE_two_numbers_equal_sum_product_quotient_l3623_362314

theorem two_numbers_equal_sum_product_quotient :
  ∃! (x y : ℝ), x ≠ 0 ∧ x + y = x * y ∧ x * y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_equal_sum_product_quotient_l3623_362314


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3623_362342

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  ((1 / (x + 1) - 1) / (x / (x^2 - 1))) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3623_362342


namespace NUMINAMATH_CALUDE_exam_score_proof_l3623_362394

/-- Given an examination with the following conditions:
  * There are 60 questions in total
  * Each correct answer scores 4 marks
  * Each wrong answer loses 1 mark
  * The total score is 130 marks
  This theorem proves that the number of correctly answered questions is 38. -/
theorem exam_score_proof (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers = 38 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l3623_362394


namespace NUMINAMATH_CALUDE_initial_snack_eaters_l3623_362321

/-- The number of snack eaters after a series of events -/
def final_snack_eaters (S : ℕ) : ℕ :=
  ((S + 20) / 2 + 10 - 30) / 2

/-- Theorem stating that the initial number of snack eaters was 100 -/
theorem initial_snack_eaters :
  ∃ S : ℕ, final_snack_eaters S = 20 ∧ S = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_snack_eaters_l3623_362321


namespace NUMINAMATH_CALUDE_bill_drew_four_pentagons_l3623_362316

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := sorry

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 88

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

theorem bill_drew_four_pentagons :
  num_pentagons = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_drew_four_pentagons_l3623_362316


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3623_362335

/-- Given a quadratic equation with coefficients a, b, and c, returns true if it has exactly one solution -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

theorem quadratic_equation_solution (b : ℝ) :
  has_one_solution 3 15 b →
  b + 3 = 36 →
  b > 3 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3623_362335


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l3623_362386

theorem triangle_is_obtuse (a b c : ℝ) (ha : a = 4) (hb : b = 6) (hc : c = 8) :
  a^2 + b^2 < c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l3623_362386


namespace NUMINAMATH_CALUDE_chocolate_triangles_l3623_362381

theorem chocolate_triangles (square_side : ℝ) (triangle_width : ℝ) (triangle_height : ℝ)
  (h_square : square_side = 10)
  (h_width : triangle_width = 1)
  (h_height : triangle_height = 3) :
  ⌊(square_side^2) / ((triangle_width * triangle_height) / 2)⌋ = 66 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_triangles_l3623_362381


namespace NUMINAMATH_CALUDE_train_speed_constant_l3623_362338

/-- A train crossing a stationary man on a platform --/
structure Train :=
  (initial_length : ℝ)
  (initial_speed : ℝ)
  (length_increase_rate : ℝ)

/-- The final speed of the train after crossing the man --/
def final_speed (t : Train) : ℝ := t.initial_speed

theorem train_speed_constant (t : Train) 
  (h1 : t.initial_length = 160)
  (h2 : t.initial_speed = 30)
  (h3 : t.length_increase_rate = 2)
  (h4 : final_speed t = t.initial_speed) :
  final_speed t = 30 := by sorry

end NUMINAMATH_CALUDE_train_speed_constant_l3623_362338


namespace NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3623_362346

def fair_12_sided_die : Finset ℕ := Finset.range 12

theorem expected_value_fair_12_sided_die : 
  (fair_12_sided_die.sum (λ x => (x + 1) * (1 : ℚ)) / 12) = (13 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3623_362346


namespace NUMINAMATH_CALUDE_customers_who_left_l3623_362300

/-- Proves that 12 customers left a waiter's section given the initial and final conditions -/
theorem customers_who_left (initial_customers : ℕ) (people_per_table : ℕ) (remaining_tables : ℕ) : 
  initial_customers = 44 → people_per_table = 8 → remaining_tables = 4 →
  initial_customers - (people_per_table * remaining_tables) = 12 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_left_l3623_362300


namespace NUMINAMATH_CALUDE_smallest_math_club_size_l3623_362374

theorem smallest_math_club_size : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (d : ℕ), d > 0 ∧ 
    (40 * n < 100 * d) ∧ 
    (100 * d < 50 * n)) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬(∃ (k : ℕ), k > 0 ∧ 
      (40 * m < 100 * k) ∧ 
      (100 * k < 50 * m))) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_math_club_size_l3623_362374


namespace NUMINAMATH_CALUDE_tan_product_less_than_one_l3623_362313

theorem tan_product_less_than_one (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  π / 2 < C ∧ C < π →  -- Angle C is obtuse
  Real.tan A * Real.tan B < 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_less_than_one_l3623_362313


namespace NUMINAMATH_CALUDE_choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l3623_362366

def num_boys : ℕ := 4
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls

-- Question 1
theorem choose_five_representatives : Nat.choose total_people 5 = 126 := by sorry

-- Question 2
theorem choose_five_with_specific_girl :
  (Nat.choose num_boys 2) * (Nat.choose (num_girls - 1) 2) = 36 := by sorry

-- Question 3
theorem choose_five_with_at_least_two_boys :
  (Nat.choose num_boys 2) * (Nat.choose num_girls 3) +
  (Nat.choose num_boys 3) * (Nat.choose num_girls 2) +
  (Nat.choose num_boys 4) * (Nat.choose num_girls 1) = 105 := by sorry

-- Question 4
theorem divide_into_three_groups :
  (Nat.choose total_people 4) * (Nat.choose (total_people - 4) 3) = 1260 := by sorry

end NUMINAMATH_CALUDE_choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l3623_362366


namespace NUMINAMATH_CALUDE_wrexham_orchestra_max_members_l3623_362356

theorem wrexham_orchestra_max_members :
  ∀ m : ℕ,
  (∃ k : ℕ, 30 * m = 31 * k + 7) →
  30 * m < 1200 →
  (∀ n : ℕ, (∃ j : ℕ, 30 * n = 31 * j + 7) → 30 * n < 1200 → 30 * n ≤ 30 * m) →
  30 * m = 720 :=
by sorry

end NUMINAMATH_CALUDE_wrexham_orchestra_max_members_l3623_362356


namespace NUMINAMATH_CALUDE_return_trip_speed_l3623_362392

/-- Given a round trip between two cities, prove the speed of the return trip -/
theorem return_trip_speed 
  (distance : ℝ) 
  (outbound_speed : ℝ) 
  (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 75 →
  average_speed = 50 →
  (2 * distance) / (distance / outbound_speed + distance / ((2 * distance) / (2 * average_speed) - distance / outbound_speed)) = average_speed →
  (2 * distance) / (2 * average_speed) - distance / outbound_speed = distance / 37.5 :=
by sorry

end NUMINAMATH_CALUDE_return_trip_speed_l3623_362392


namespace NUMINAMATH_CALUDE_largest_divisor_of_60_36_divisible_by_3_l3623_362323

theorem largest_divisor_of_60_36_divisible_by_3 :
  ∃ (n : ℕ), n > 0 ∧ n ∣ 60 ∧ n ∣ 36 ∧ 3 ∣ n ∧
  ∀ (m : ℕ), m > n → (m ∣ 60 ∧ m ∣ 36 ∧ 3 ∣ m) → False :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_60_36_divisible_by_3_l3623_362323


namespace NUMINAMATH_CALUDE_linda_savings_l3623_362393

theorem linda_savings (savings : ℝ) : 
  savings > 0 →
  savings * (3/4) + savings * (1/8) + 250 = savings * (7/8) →
  250 = (savings * (1/8)) * 0.9 →
  savings = 2222.24 := by
sorry

end NUMINAMATH_CALUDE_linda_savings_l3623_362393


namespace NUMINAMATH_CALUDE_light_reflection_l3623_362322

/-- A beam of light passing through a point and reflecting off a circle --/
structure LightBeam where
  M : ℝ × ℝ
  C : Set (ℝ × ℝ)

/-- Definition of the circle C --/
def is_circle (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 7)^2 = 25

/-- Definition of the reflected light ray equation --/
def reflected_ray_equation (x y : ℝ) : Prop :=
  x + y - 7 = 0

/-- Definition of the range of incident point A --/
def incident_point_range (A : ℝ) : Prop :=
  1 ≤ A ∧ A ≤ 23/2

/-- Main theorem --/
theorem light_reflection (beam : LightBeam) 
  (h_M : beam.M = (25, 18))
  (h_C : is_circle beam.C) :
  (∀ x y, reflected_ray_equation x y ↔ 
    (x, y) ∈ {p | ∃ t, p = ((1-t) * 25 + t * 0, (1-t) * (-18) + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) ∧
  (∀ A, incident_point_range A ↔ 
    ∃ (k : ℝ), (A, 0) ∈ {p | ∃ t, p = ((1-t) * 25 + t * A, (1-t) * (-18) + t * 0) ∧ 0 ≤ t ∧ t ≤ 1} ∧
               (0, 7) ∈ {p | ∃ t, p = ((1-t) * A + t * 0, (1-t) * 0 + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) :=
sorry

end NUMINAMATH_CALUDE_light_reflection_l3623_362322


namespace NUMINAMATH_CALUDE_average_fish_caught_l3623_362339

def aang_fish : ℕ := 7
def sokka_fish : ℕ := 5
def toph_fish : ℕ := 12
def total_people : ℕ := 3

theorem average_fish_caught :
  (aang_fish + sokka_fish + toph_fish) / total_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_caught_l3623_362339


namespace NUMINAMATH_CALUDE_train_speed_l3623_362350

-- Define the train's parameters
def train_length : Real := 240  -- in meters
def crossing_time : Real := 16  -- in seconds

-- Define the conversion factor from m/s to km/h
def mps_to_kmh : Real := 3.6

-- Theorem statement
theorem train_speed :
  let speed_mps := train_length / crossing_time
  let speed_kmh := speed_mps * mps_to_kmh
  speed_kmh = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3623_362350


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3623_362365

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 3 * a 5 = 16) :
  ∃ q : ℝ, (q = Real.sqrt 2 ∨ q = Real.sqrt 2 / 2) ∧ 
    ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3623_362365


namespace NUMINAMATH_CALUDE_radius_of_equal_area_circle_radius_of_equal_area_circle_specific_l3623_362389

/-- The radius of a circle with area equal to the difference between two concentric circles -/
theorem radius_of_equal_area_circle (r₁ r₂ : ℝ) (h : 0 < r₁ ∧ r₁ < r₂) :
  ∃ r : ℝ, r > 0 ∧ π * r^2 = π * (r₂^2 - r₁^2) :=
by
  sorry

/-- The specific case where r₁ = 15 and r₂ = 25 -/
theorem radius_of_equal_area_circle_specific :
  ∃ r : ℝ, r > 0 ∧ π * r^2 = π * (25^2 - 15^2) ∧ r = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_radius_of_equal_area_circle_radius_of_equal_area_circle_specific_l3623_362389


namespace NUMINAMATH_CALUDE_ones_digit_of_36_power_ones_digit_of_36_to_large_power_l3623_362312

theorem ones_digit_of_36_power (n : ℕ) : (36 ^ n) % 10 = 6 := by sorry

theorem ones_digit_of_36_to_large_power :
  (36 ^ (36 * (5 ^ 5))) % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_36_power_ones_digit_of_36_to_large_power_l3623_362312


namespace NUMINAMATH_CALUDE_sum_of_constants_l3623_362306

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a + b / x^2) →
  (2 = a + b) →
  (6 = a + b / 9) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3623_362306


namespace NUMINAMATH_CALUDE_largest_number_problem_l3623_362369

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 3) :
  c = 33.25 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3623_362369


namespace NUMINAMATH_CALUDE_inequality_proof_l3623_362317

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3623_362317


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l3623_362395

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 18 → x + y ≤ Real.sqrt 166 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l3623_362395


namespace NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3623_362363

-- Define the cube dimensions
def large_cube_edge : ℕ := 4
def small_cube_edge : ℕ := 1

-- Define the number of cubes
def total_cubes : ℕ := 64
def blue_cubes : ℕ := 48
def yellow_cubes : ℕ := 16

-- Define the surface area of the large cube
def large_cube_surface_area : ℕ := 6 * large_cube_edge * large_cube_edge

-- Define the minimum number of yellow cubes that must be on the surface
def min_yellow_surface_cubes : ℕ := yellow_cubes - 1

-- Theorem statement
theorem min_yellow_surface_fraction :
  (min_yellow_surface_cubes : ℚ) / large_cube_surface_area = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3623_362363


namespace NUMINAMATH_CALUDE_qq_fish_tank_theorem_l3623_362398

/-- Represents the fish tank scenario --/
structure FishTank where
  total_fish : Nat
  blue_fish : Nat
  black_fish : Nat
  daily_catch : Nat

/-- Probability that Mr. QQ eats at least a certain number of fish --/
def prob_eat_at_least (tank : FishTank) (n : Nat) : ℚ :=
  sorry

/-- Expected number of fish eaten by Mr. QQ --/
def expected_fish_eaten (tank : FishTank) : ℚ :=
  sorry

/-- The main theorem about Mr. QQ's fish tank --/
theorem qq_fish_tank_theorem (tank : FishTank) 
    (h1 : tank.total_fish = 7)
    (h2 : tank.blue_fish = 6)
    (h3 : tank.black_fish = 1)
    (h4 : tank.daily_catch = 1) :
  (prob_eat_at_least tank 5 = 19/35) ∧ 
  (expected_fish_eaten tank = 5) :=
by sorry

end NUMINAMATH_CALUDE_qq_fish_tank_theorem_l3623_362398


namespace NUMINAMATH_CALUDE_parade_team_size_l3623_362373

theorem parade_team_size : 
  ∃ n : ℕ, 
    n % 5 = 0 ∧ 
    n ≥ 1000 ∧ 
    n % 4 = 3 ∧ 
    n % 3 = 2 ∧ 
    n % 2 = 1 ∧ 
    n = 1045 ∧ 
    ∀ m : ℕ, 
      (m % 5 = 0 ∧ 
       m ≥ 1000 ∧ 
       m % 4 = 3 ∧ 
       m % 3 = 2 ∧ 
       m % 2 = 1) → 
      m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_parade_team_size_l3623_362373


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3623_362308

/-- The equation of the tangent line to y = 2x - x³ at (1, 1) is x + y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = 2*x - x^3) → -- The curve equation
  ((1 : ℝ) = 1 → (2*(1 : ℝ) - (1 : ℝ)^3) = 1) → -- The point (1, 1) lies on the curve
  (x + y - 2 = 0) ↔ -- The tangent line equation
  (∃ (m : ℝ), y - 1 = m * (x - 1) ∧ 
              m = (2 - 3*(1 : ℝ)^2)) -- Slope of the tangent line at x = 1
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3623_362308


namespace NUMINAMATH_CALUDE_product_calculation_l3623_362352

theorem product_calculation : 12 * 0.2 * 3 * 0.1 / 0.6 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l3623_362352


namespace NUMINAMATH_CALUDE_soccer_match_ratio_l3623_362360

def soccer_match (kickers_second_period : ℕ) : Prop :=
  let kickers_first_period : ℕ := 2
  let spiders_first_period : ℕ := kickers_first_period / 2
  let spiders_second_period : ℕ := 2 * kickers_second_period
  let total_goals : ℕ := 15
  (kickers_first_period + kickers_second_period + spiders_first_period + spiders_second_period = total_goals) ∧
  (kickers_second_period : ℚ) / (kickers_first_period : ℚ) = 2 / 1

theorem soccer_match_ratio : ∃ (kickers_second_period : ℕ), soccer_match kickers_second_period := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_ratio_l3623_362360


namespace NUMINAMATH_CALUDE_min_distance_PM_l3623_362391

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define a point P on l₁
structure Point_P where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y

-- Define a line l₂ passing through P
structure Line_l₂ (P : Point_P) where
  slope : ℝ
  passes_through_P : True  -- This is a simplification, as we don't need the specific equation

-- Define the intersection point M
structure Point_M (P : Point_P) (l₂ : Line_l₂ P) where
  x : ℝ
  y : ℝ
  on_C : C x y
  on_l₂ : True  -- This is a simplification, as we don't need the specific condition

-- State the theorem
theorem min_distance_PM (P : Point_P) (l₂ : Line_l₂ P) (M : Point_M P l₂) :
  ∃ (d : ℝ), d = 4 ∧ ∀ (M' : Point_M P l₂), Real.sqrt ((M'.x - P.x)^2 + (M'.y - P.y)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_PM_l3623_362391


namespace NUMINAMATH_CALUDE_felix_brother_lifting_capacity_l3623_362353

/-- Given information about Felix and his brother's weights and lifting capacities,
    prove how much Felix's brother can lift off the ground. -/
theorem felix_brother_lifting_capacity
  (felix_lift_ratio : ℝ)
  (felix_lift_weight : ℝ)
  (brother_weight_ratio : ℝ)
  (brother_lift_ratio : ℝ)
  (h1 : felix_lift_ratio = 1.5)
  (h2 : felix_lift_weight = 150)
  (h3 : brother_weight_ratio = 2)
  (h4 : brother_lift_ratio = 3) :
  felix_lift_weight * brother_weight_ratio * brother_lift_ratio / felix_lift_ratio = 600 :=
by sorry

end NUMINAMATH_CALUDE_felix_brother_lifting_capacity_l3623_362353


namespace NUMINAMATH_CALUDE_brown_dogs_l3623_362327

def kennel (total : ℕ) (long_fur : ℕ) (neither : ℕ) : Prop :=
  total = 45 ∧
  long_fur = 36 ∧
  neither = 8 ∧
  long_fur ≤ total ∧
  neither ≤ total - long_fur

theorem brown_dogs (total long_fur neither : ℕ) 
  (h : kennel total long_fur neither) : ∃ brown : ℕ, brown = 37 :=
sorry

end NUMINAMATH_CALUDE_brown_dogs_l3623_362327


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l3623_362302

/-- Rainfall data for a week --/
structure RainfallData where
  monday_morning : ℝ
  monday_afternoon : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  daily_average : ℝ
  num_days : ℕ

/-- Theorem stating the ratio of Tuesday's rainfall to Monday's total rainfall --/
theorem tuesday_to_monday_ratio (data : RainfallData) : 
  data.monday_morning = 2 ∧ 
  data.monday_afternoon = 1 ∧ 
  data.wednesday = 0 ∧ 
  data.thursday = 1 ∧ 
  data.friday = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday ∧
  data.daily_average = 4 ∧
  data.num_days = 5 ∧
  data.daily_average * data.num_days = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday + data.friday →
  data.tuesday / (data.monday_morning + data.monday_afternoon) = 2 := by
sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l3623_362302


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3623_362396

theorem roots_of_quadratic (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = 6 ∧ x₁ * x₂ = -7 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3623_362396


namespace NUMINAMATH_CALUDE_sum_base4_equals_l3623_362343

/-- Converts a base 4 number (represented as a list of digits) to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits). -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 4) ((m % 4) :: acc)
    go n []

/-- The theorem to be proved -/
theorem sum_base4_equals : 
  natToBase4 (base4ToNat [3, 0, 2] + base4ToNat [2, 2, 1] + 
              base4ToNat [1, 3, 2] + base4ToNat [0, 1, 1]) = [3, 3, 2, 2] := by
  sorry


end NUMINAMATH_CALUDE_sum_base4_equals_l3623_362343


namespace NUMINAMATH_CALUDE_gym_towels_theorem_l3623_362310

/-- Represents the number of guests entering the gym each hour -/
structure GymHours :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of towels used based on gym hours -/
def totalTowels (hours : GymHours) : ℕ :=
  hours.first + hours.second + hours.third + hours.fourth

/-- Theorem: Given the specified conditions, the total number of towels used is 285 -/
theorem gym_towels_theorem (hours : GymHours) 
  (h1 : hours.first = 50)
  (h2 : hours.second = hours.first + hours.first / 5)
  (h3 : hours.third = hours.second + hours.second / 4)
  (h4 : hours.fourth = hours.third + hours.third / 3)
  : totalTowels hours = 285 := by
  sorry

#eval totalTowels { first := 50, second := 60, third := 75, fourth := 100 }

end NUMINAMATH_CALUDE_gym_towels_theorem_l3623_362310


namespace NUMINAMATH_CALUDE_only_point_A_in_region_l3623_362340

def plane_region (x y : ℝ) : Prop := x + y - 1 < 0

def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (2, 4)
def point_C : ℝ × ℝ := (-1, 4)
def point_D : ℝ × ℝ := (1, 8)

theorem only_point_A_in_region :
  plane_region point_A.1 point_A.2 ∧
  ¬plane_region point_B.1 point_B.2 ∧
  ¬plane_region point_C.1 point_C.2 ∧
  ¬plane_region point_D.1 point_D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_point_A_in_region_l3623_362340


namespace NUMINAMATH_CALUDE_central_circle_radius_l3623_362307

/-- The radius of a circle tangent to six semicircles evenly arranged inside a regular hexagon -/
theorem central_circle_radius (side_length : ℝ) (h : side_length = 3) :
  let apothem := side_length * (Real.sqrt 3 / 2)
  let semicircle_radius := side_length / 2
  let central_radius := apothem - semicircle_radius
  central_radius = (3 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_central_circle_radius_l3623_362307


namespace NUMINAMATH_CALUDE_top_face_after_16_rounds_l3623_362315

/-- Represents the faces of a cube -/
inductive Face : Type
  | A | B | C | D | E | F

/-- Represents the state of the cube -/
structure CubeState :=
  (top : Face)
  (front : Face)
  (right : Face)
  (back : Face)
  (left : Face)
  (bottom : Face)

/-- Performs one round of operations on the cube -/
def perform_round (state : CubeState) : CubeState :=
  sorry

/-- Initial state of the cube -/
def initial_state : CubeState :=
  { top := Face.E,
    front := Face.A,
    right := Face.C,
    back := Face.B,
    left := Face.D,
    bottom := Face.F }

/-- Theorem stating that after 16 rounds, the top face will be E -/
theorem top_face_after_16_rounds (n : Nat) :
  (n = 16) → (perform_round^[n] initial_state).top = Face.E :=
sorry

end NUMINAMATH_CALUDE_top_face_after_16_rounds_l3623_362315


namespace NUMINAMATH_CALUDE_eulers_formula_l3623_362331

/-- A closed polyhedron is a structure with a number of edges, faces, and vertices. -/
structure ClosedPolyhedron where
  edges : ℕ
  faces : ℕ
  vertices : ℕ

/-- Euler's formula for polyhedra states that for any closed polyhedron, 
    the number of edges plus 2 is equal to the sum of the number of faces and vertices. -/
theorem eulers_formula (p : ClosedPolyhedron) : p.edges + 2 = p.faces + p.vertices := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l3623_362331
