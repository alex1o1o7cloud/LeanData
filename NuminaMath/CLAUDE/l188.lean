import Mathlib

namespace NUMINAMATH_CALUDE_convex_polygon_27_sides_diagonals_l188_18842

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 27 sides has 324 diagonals -/
theorem convex_polygon_27_sides_diagonals :
  num_diagonals 27 = 324 := by sorry

end NUMINAMATH_CALUDE_convex_polygon_27_sides_diagonals_l188_18842


namespace NUMINAMATH_CALUDE_equation_solution_l188_18824

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l188_18824


namespace NUMINAMATH_CALUDE_john_reading_days_l188_18878

/-- Given that John reads 4 books a day and 48 books in 6 weeks, prove that he reads on 2 days per week. -/
theorem john_reading_days 
  (books_per_day : ℕ) 
  (total_books : ℕ) 
  (total_weeks : ℕ) 
  (h1 : books_per_day = 4) 
  (h2 : total_books = 48) 
  (h3 : total_weeks = 6) : 
  (total_books / books_per_day) / total_weeks = 2 :=
by sorry

end NUMINAMATH_CALUDE_john_reading_days_l188_18878


namespace NUMINAMATH_CALUDE_power_five_minus_self_divisible_by_five_l188_18815

theorem power_five_minus_self_divisible_by_five (a : ℤ) : ∃ k : ℤ, a^5 - a = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_power_five_minus_self_divisible_by_five_l188_18815


namespace NUMINAMATH_CALUDE_first_group_size_l188_18889

/-- The number of men in the first group -/
def M : ℕ := 42

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def men_second_group : ℕ := 27

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 28

/-- The work done by a group is inversely proportional to the number of days they take -/
axiom work_inverse_proportion (men days : ℕ) : men * days = men_second_group * days_second_group

theorem first_group_size : M = 42 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l188_18889


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l188_18893

/-- Given a sinusoidal function y = a * sin(bx + c) + d with positive constants a, b, c, and d,
    if the function oscillates between 5 and -3, then a = 4 -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l188_18893


namespace NUMINAMATH_CALUDE_odd_integers_between_9_and_39_l188_18832

theorem odd_integers_between_9_and_39 :
  let first_term := 9
  let last_term := 39
  let sum := 384
  let n := (last_term - first_term) / 2 + 1
  n = 16 ∧ sum = n / 2 * (first_term + last_term) := by
sorry

end NUMINAMATH_CALUDE_odd_integers_between_9_and_39_l188_18832


namespace NUMINAMATH_CALUDE_class_average_problem_l188_18860

theorem class_average_problem (group1_percent : Real) (group1_score : Real)
                              (group2_percent : Real)
                              (group3_percent : Real) (group3_score : Real)
                              (total_average : Real) :
  group1_percent = 0.25 →
  group1_score = 0.8 →
  group2_percent = 0.5 →
  group3_percent = 0.25 →
  group3_score = 0.9 →
  total_average = 0.75 →
  group1_percent + group2_percent + group3_percent = 1 →
  group1_percent * group1_score + group2_percent * (65 / 100) + group3_percent * group3_score = total_average :=
by
  sorry


end NUMINAMATH_CALUDE_class_average_problem_l188_18860


namespace NUMINAMATH_CALUDE_shaded_percentage_of_square_l188_18876

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  bottomLeft : Point

/-- Represents a shaded region -/
structure ShadedRegion where
  bottomLeft : Point
  topRight : Point

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.sideLength * s.sideLength

/-- Calculate the area of a shaded region -/
def shadedRegionArea (r : ShadedRegion) : ℝ :=
  (r.topRight.x - r.bottomLeft.x) * (r.topRight.y - r.bottomLeft.y)

/-- The main theorem -/
theorem shaded_percentage_of_square (EFGH : Square)
  (region1 region2 region3 : ShadedRegion) :
  EFGH.sideLength = 7 →
  EFGH.bottomLeft = ⟨0, 0⟩ →
  region1 = ⟨⟨0, 0⟩, ⟨1, 1⟩⟩ →
  region2 = ⟨⟨3, 0⟩, ⟨5, 5⟩⟩ →
  region3 = ⟨⟨6, 0⟩, ⟨7, 7⟩⟩ →
  (shadedRegionArea region1 + shadedRegionArea region2 + shadedRegionArea region3) /
    squareArea EFGH * 100 = 14 / 49 * 100 := by
  sorry

end NUMINAMATH_CALUDE_shaded_percentage_of_square_l188_18876


namespace NUMINAMATH_CALUDE_batsman_average_l188_18852

def average (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average (totalRuns18 : ℕ) (totalRuns17 : ℕ) :
  average totalRuns18 18 = 18 →
  totalRuns18 = totalRuns17 + 1 →
  average totalRuns17 17 = 19 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l188_18852


namespace NUMINAMATH_CALUDE_power_of_power_equals_power_product_l188_18847

theorem power_of_power_equals_power_product (x : ℝ) : (x^2)^4 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_power_product_l188_18847


namespace NUMINAMATH_CALUDE_cone_slant_height_l188_18836

/-- Given a cone with base radius 3 cm and curved surface area 141.3716694115407 cm²,
    prove that its slant height is 15 cm. -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 3) (h2 : csa = 141.3716694115407) :
  csa / (Real.pi * r) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l188_18836


namespace NUMINAMATH_CALUDE_chris_mixture_problem_l188_18851

/-- Given the conditions of Chris's mixture of raisins and nuts, prove that the number of pounds of nuts is 4. -/
theorem chris_mixture_problem (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) (nut_cost : ℝ) :
  raisin_pounds = 3 →
  nut_cost = 2 * raisin_cost →
  (raisin_pounds * raisin_cost) = (3 / 11) * (raisin_pounds * raisin_cost + nut_pounds * nut_cost) →
  nut_pounds = 4 := by
sorry

end NUMINAMATH_CALUDE_chris_mixture_problem_l188_18851


namespace NUMINAMATH_CALUDE_pencil_difference_l188_18849

theorem pencil_difference (price : ℚ) (jamar_count sharona_count : ℕ) : 
  price > 0.01 →
  price * jamar_count = 216/100 →
  price * sharona_count = 272/100 →
  sharona_count - jamar_count = 7 := by
sorry

end NUMINAMATH_CALUDE_pencil_difference_l188_18849


namespace NUMINAMATH_CALUDE_fewest_tiles_needed_l188_18873

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a single tile -/
def tileDimensions : Dimensions := ⟨6, 2⟩

/-- The dimensions of the rectangular region in feet -/
def regionDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the rectangular region in inches -/
def regionDimensionsInches : Dimensions :=
  ⟨feetToInches regionDimensionsFeet.length, feetToInches regionDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover a given area -/
def tilesNeeded (regionArea tileArea : ℕ) : ℕ :=
  (regionArea + tileArea - 1) / tileArea

theorem fewest_tiles_needed :
  tilesNeeded (area regionDimensionsInches) (area tileDimensions) = 216 := by
  sorry

end NUMINAMATH_CALUDE_fewest_tiles_needed_l188_18873


namespace NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l188_18871

theorem range_of_m_for_false_proposition : 
  (∃ m : ℝ, ¬(∀ x : ℝ, x^2 - 2*x - m ≥ 0)) ↔ 
  (∃ m : ℝ, m > -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l188_18871


namespace NUMINAMATH_CALUDE_election_probabilities_l188_18809

theorem election_probabilities 
  (pA pB pC : ℝ)
  (hA : pA = 4/5)
  (hB : pB = 3/5)
  (hC : pC = 7/10) :
  let p_exactly_one := pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC
  let p_at_most_two := 1 - pA * pB * pC
  (p_exactly_one = 47/250) ∧ (p_at_most_two = 83/125) := by
  sorry

end NUMINAMATH_CALUDE_election_probabilities_l188_18809


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l188_18828

theorem complex_fraction_equals_neg_i : (1 - I) / (1 + I) = -I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l188_18828


namespace NUMINAMATH_CALUDE_john_account_balance_l188_18841

/-- Calculates the final balance after a deposit and withdrawal -/
def final_balance (initial_balance deposit withdrawal : ℚ) : ℚ :=
  initial_balance + deposit - withdrawal

/-- Theorem: Given the specified initial balance, deposit, and withdrawal,
    the final balance is 43.8 -/
theorem john_account_balance :
  final_balance 45.7 18.6 20.5 = 43.8 := by
  sorry

end NUMINAMATH_CALUDE_john_account_balance_l188_18841


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l188_18801

theorem min_value_quadratic_form (x₁ x₂ x₃ x₄ : ℝ) 
  (h : 5*x₁ + 6*x₂ - 7*x₃ + 4*x₄ = 1) : 
  3*x₁^2 + 2*x₂^2 + 5*x₃^2 + x₄^2 ≥ 15/782 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l188_18801


namespace NUMINAMATH_CALUDE_johns_family_members_l188_18834

/-- The number of family members on John's father's side -/
def fathers_side : ℕ := sorry

/-- The total number of family members -/
def total_members : ℕ := 23

/-- The ratio of mother's side to father's side -/
def mother_ratio : ℚ := 13/10

theorem johns_family_members :
  fathers_side = 10 ∧
  (fathers_side : ℚ) * (1 + mother_ratio - 1) + fathers_side = total_members :=
sorry

end NUMINAMATH_CALUDE_johns_family_members_l188_18834


namespace NUMINAMATH_CALUDE_parallel_lines_a_eq_neg_one_l188_18897

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line ax + by + c = 0 is -a/b when b ≠ 0 -/
axiom line_slope {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -a/b * x - c/b

theorem parallel_lines_a_eq_neg_one (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ x + (a - 1) * y + a^2 - 1 = 0) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_eq_neg_one_l188_18897


namespace NUMINAMATH_CALUDE_union_of_sets_l188_18806

-- Define the sets A and B
def A (a : ℤ) : Set ℤ := {|a + 1|, 3, 5}
def B (a : ℤ) : Set ℤ := {2 * a + 1, a^2 + 2 * a, a^2 + 2 * a - 1}

-- Define the theorem
theorem union_of_sets :
  ∃ a : ℤ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l188_18806


namespace NUMINAMATH_CALUDE_equal_area_segment_property_l188_18880

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  base_difference : longer_base = shorter_base + 150
  midpoint_ratio : ℝ
  midpoint_area_ratio : midpoint_ratio = 3 / 4

/-- The length of the segment that divides the trapezoid into two equal-area regions -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  t.shorter_base + 150

/-- Theorem stating the property of the equal area segment -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^3 / 1000⌋ = 142 := by
  sorry

#check equal_area_segment_property

end NUMINAMATH_CALUDE_equal_area_segment_property_l188_18880


namespace NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l188_18885

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) :
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = Complex.exp (Real.pi * Complex.I / 4) * z₁ →
  a^2 / b = 4 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_isosceles_triangle_l188_18885


namespace NUMINAMATH_CALUDE_donation_problem_l188_18898

/-- Calculates the total number of articles of clothing donated given the initial number of items set aside by Adam and the number of friends donating. -/
def total_donated_clothing (adam_pants : ℕ) (adam_jumpers : ℕ) (adam_pajama_sets : ℕ) (adam_tshirts : ℕ) (num_friends : ℕ) : ℕ := 
  let adam_initial := adam_pants + adam_jumpers + (2 * adam_pajama_sets) + adam_tshirts
  let friends_donation := num_friends * adam_initial
  let adam_final := adam_initial / 2
  adam_final + friends_donation

/-- Theorem stating that the total number of articles of clothing donated is 126, given the specific conditions of the problem. -/
theorem donation_problem : total_donated_clothing 4 4 4 20 3 = 126 := by
  sorry

end NUMINAMATH_CALUDE_donation_problem_l188_18898


namespace NUMINAMATH_CALUDE_carpet_coverage_percentage_l188_18817

/-- The percentage of a living room floor covered by a rectangular carpet -/
theorem carpet_coverage_percentage 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (room_area : ℝ) 
  (h1 : carpet_length = 4) 
  (h2 : carpet_width = 9) 
  (h3 : room_area = 120) : 
  (carpet_length * carpet_width) / room_area * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_carpet_coverage_percentage_l188_18817


namespace NUMINAMATH_CALUDE_number_of_type_C_is_16_l188_18818

/-- Represents the types of people in the problem -/
inductive PersonType
| A
| B
| C

/-- The total number of people -/
def total_people : ℕ := 25

/-- The number of people who answered "yes" to "Are you a Type A person?" -/
def yes_to_A : ℕ := 17

/-- The number of people who answered "yes" to "Are you a Type C person?" -/
def yes_to_C : ℕ := 12

/-- The number of people who answered "yes" to "Are you a Type B person?" -/
def yes_to_B : ℕ := 8

/-- Theorem stating that the number of Type C people is 16 -/
theorem number_of_type_C_is_16 :
  ∃ (a b c : ℕ),
    a + b + c = total_people ∧
    a + b + (c / 2) = yes_to_A ∧
    b + (c / 2) = yes_to_C ∧
    c / 2 = yes_to_B ∧
    c = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_type_C_is_16_l188_18818


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l188_18883

theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → (m + 3) / x₁ < (m + 3) / x₂) → 
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l188_18883


namespace NUMINAMATH_CALUDE_almond_croissant_price_l188_18821

def white_bread_price : ℝ := 3.50
def baguette_price : ℝ := 1.50
def sourdough_price : ℝ := 4.50
def total_spent : ℝ := 78.00
def num_weeks : ℕ := 4

def weekly_bread_cost : ℝ := 2 * white_bread_price + baguette_price + 2 * sourdough_price

theorem almond_croissant_price :
  ∃ (croissant_price : ℝ),
    croissant_price * num_weeks + weekly_bread_cost * num_weeks = total_spent ∧
    croissant_price = 8.00 := by
  sorry

end NUMINAMATH_CALUDE_almond_croissant_price_l188_18821


namespace NUMINAMATH_CALUDE_exactly_two_transformations_map_pattern_l188_18819

/-- A pattern on a line consisting of alternating right-facing and left-facing triangles,
    followed by their vertically flipped versions, creating a symmetric, infinite, repeating pattern. -/
structure TrianglePattern where
  ℓ : Line

/-- Transformations that can be applied to the pattern -/
inductive Transformation
  | Rotate90 : Point → Transformation
  | TranslateParallel : Real → Transformation
  | Rotate120 : Point → Transformation
  | TranslatePerpendicular : Real → Transformation

/-- Predicate to check if a transformation maps the pattern onto itself -/
def maps_onto_self (t : Transformation) (p : TrianglePattern) : Prop :=
  sorry

theorem exactly_two_transformations_map_pattern (p : TrianglePattern) :
  ∃! (ts : Finset Transformation), ts.card = 2 ∧
    (∀ t ∈ ts, maps_onto_self t p) ∧
    (∀ t : Transformation, maps_onto_self t p → t ∈ ts) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_transformations_map_pattern_l188_18819


namespace NUMINAMATH_CALUDE_polygon_sides_diagonals_l188_18816

theorem polygon_sides_diagonals : ∃ (n : ℕ), n > 2 ∧ 3 * n * (n * (n - 3)) = 300 := by
  use 10
  sorry

end NUMINAMATH_CALUDE_polygon_sides_diagonals_l188_18816


namespace NUMINAMATH_CALUDE_special_factorization_of_630_l188_18877

theorem special_factorization_of_630 : ∃ (a b x y z : ℕ), 
  (a + 1 = b) ∧ 
  (x + 1 = y) ∧ 
  (y + 1 = z) ∧ 
  (a * b = 630) ∧ 
  (x * y * z = 630) ∧ 
  (a + b + x + y + z = 75) := by
  sorry

end NUMINAMATH_CALUDE_special_factorization_of_630_l188_18877


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l188_18872

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.36 * S^2) :
  R = 0.6 * S := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l188_18872


namespace NUMINAMATH_CALUDE_unique_positive_solution_l188_18803

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l188_18803


namespace NUMINAMATH_CALUDE_distance_between_points_on_line_l188_18846

/-- Given a line with equation 2x - 3y + 6 = 0 and two points (p, q) and (r, s) on this line,
    the distance between these points is (√13/3)|r - p| -/
theorem distance_between_points_on_line (p r : ℝ) :
  let q := (2*p + 6)/3
  let s := (2*r + 6)/3
  (2*p - 3*q + 6 = 0) →
  (2*r - 3*s + 6 = 0) →
  Real.sqrt ((r - p)^2 + (s - q)^2) = (Real.sqrt 13 / 3) * |r - p| := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_on_line_l188_18846


namespace NUMINAMATH_CALUDE_complement_A_eq_three_four_l188_18869

-- Define the set A
def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Define the complement of A with respect to ℕ
def complement_A : Set ℕ := {x : ℕ | x ∉ A}

-- Theorem statement
theorem complement_A_eq_three_four : complement_A = {3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_eq_three_four_l188_18869


namespace NUMINAMATH_CALUDE_strategy_game_cost_l188_18886

/-- The cost of Tom's video game purchases -/
def total_cost : ℚ := 35.52

/-- The cost of the football game -/
def football_cost : ℚ := 14.02

/-- The cost of the Batman game -/
def batman_cost : ℚ := 12.04

/-- The cost of the strategy game -/
def strategy_cost : ℚ := total_cost - football_cost - batman_cost

theorem strategy_game_cost :
  strategy_cost = 9.46 := by sorry

end NUMINAMATH_CALUDE_strategy_game_cost_l188_18886


namespace NUMINAMATH_CALUDE_amiths_age_l188_18862

theorem amiths_age (a d : ℕ) : 
  (a - 5 = 3 * (d - 5)) → 
  (a + 10 = 2 * (d + 10)) → 
  a = 50 := by
sorry

end NUMINAMATH_CALUDE_amiths_age_l188_18862


namespace NUMINAMATH_CALUDE_division_remainder_l188_18870

theorem division_remainder (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  ∃ (q : ℕ), 2 * x + 3 * u * y = q * y + (if 2 * v < y then 2 * v else 2 * v - y) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l188_18870


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l188_18856

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ a b : ℝ, (a^2 + b^2 = 16) ∧ 
              (x = (10 + a) / 2) ∧ 
              (y = b / 2)) → 
  (x - 5)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l188_18856


namespace NUMINAMATH_CALUDE_walter_work_hours_l188_18845

/-- Walter's work schedule and earnings -/
structure WorkSchedule where
  days_per_week : ℕ
  hourly_rate : ℚ
  allocation_ratio : ℚ
  school_allocation : ℚ

/-- Calculate the daily work hours given a work schedule -/
def daily_work_hours (schedule : WorkSchedule) : ℚ :=
  schedule.school_allocation / (schedule.days_per_week * schedule.hourly_rate * schedule.allocation_ratio)

/-- Theorem: Walter works 4 hours a day -/
theorem walter_work_hours : 
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hourly_rate := 5,
    allocation_ratio := 3/4,
    school_allocation := 75
  }
  daily_work_hours walter_schedule = 4 := by
  sorry

end NUMINAMATH_CALUDE_walter_work_hours_l188_18845


namespace NUMINAMATH_CALUDE_mary_minus_robert_eq_two_l188_18823

/-- Represents the candy distribution problem -/
structure CandyDistribution where
  total : Nat
  kate : Nat
  robert : Nat
  bill : Nat
  mary : Nat
  kate_pieces : kate = 4
  robert_more_than_kate : robert = kate + 2
  bill_less_than_mary : bill + 6 = mary
  kate_more_than_bill : kate = bill + 2
  mary_more_than_robert : mary > robert

/-- Proves that Mary gets 2 more pieces of candy than Robert -/
theorem mary_minus_robert_eq_two (cd : CandyDistribution) : cd.mary - cd.robert = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_minus_robert_eq_two_l188_18823


namespace NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l188_18853

theorem xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two :
  ∀ x y : ℝ,
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_minus_x_squared_y_equals_negative_two_sqrt_two_l188_18853


namespace NUMINAMATH_CALUDE_rectangular_field_area_l188_18867

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area (uncovered_side : ℝ) (total_fencing : ℝ) : uncovered_side = 20 → total_fencing = 76 → uncovered_side * ((total_fencing - uncovered_side) / 2) = 560 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l188_18867


namespace NUMINAMATH_CALUDE_luca_pizza_ingredients_l188_18829

/-- Calculates the required amount of milk and oil for a given amount of flour in Luca's pizza dough recipe. -/
def pizza_ingredients (flour : ℚ) : ℚ × ℚ :=
  let milk_ratio : ℚ := 70 / 350
  let oil_ratio : ℚ := 30 / 350
  (flour * milk_ratio, flour * oil_ratio)

/-- Proves that for 1050 mL of flour, Luca needs 210 mL of milk and 90 mL of oil. -/
theorem luca_pizza_ingredients : pizza_ingredients 1050 = (210, 90) := by
  sorry

end NUMINAMATH_CALUDE_luca_pizza_ingredients_l188_18829


namespace NUMINAMATH_CALUDE_expression_value_l188_18857

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_l188_18857


namespace NUMINAMATH_CALUDE_simplify_expression_l188_18875

theorem simplify_expression (w : ℝ) :
  2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l188_18875


namespace NUMINAMATH_CALUDE_y_divisibility_l188_18813

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  (∃ k : ℕ, y = 40 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l188_18813


namespace NUMINAMATH_CALUDE_swimmers_pass_21_times_l188_18839

/-- Represents the swimming pool setup and swimmer characteristics --/
structure SwimmingSetup where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (setup : SwimmingSetup) : ℕ :=
  sorry

/-- Theorem stating that the swimmers pass each other 21 times --/
theorem swimmers_pass_21_times :
  let setup : SwimmingSetup := {
    poolLength := 120,
    swimmerASpeed := 4,
    swimmerBSpeed := 3,
    totalTime := 15 * 60  -- 15 minutes in seconds
  }
  calculatePassings setup = 21 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_21_times_l188_18839


namespace NUMINAMATH_CALUDE_angle_A_measure_l188_18843

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- State the theorem
theorem angle_A_measure (t : Triangle) 
  (h1 : t.C = 3 * t.B) 
  (h2 : t.B = 15) 
  (h3 : t.A + t.B + t.C = 180) : 
  t.A = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l188_18843


namespace NUMINAMATH_CALUDE_first_game_score_l188_18807

def basketball_scores : List ℕ := [68, 70, 61, 74, 62, 65, 74]

theorem first_game_score (mean : ℚ) (h1 : mean = 67.9) :
  ∃ x : ℕ, (x :: basketball_scores).length = 8 ∧ 
  (((x :: basketball_scores).sum : ℚ) / 8 = mean) ∧
  x = 69 := by
  sorry

end NUMINAMATH_CALUDE_first_game_score_l188_18807


namespace NUMINAMATH_CALUDE_circular_track_length_circular_track_length_is_280_l188_18800

/-- The length of a circular track given specific running conditions -/
theorem circular_track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed jim_speed : ℝ),
      brenda_speed > 0 ∧ jim_speed > 0 →
      ∃ (first_meet_time second_meet_time : ℝ),
        first_meet_time > 0 ∧ second_meet_time > first_meet_time ∧
        brenda_speed * first_meet_time = 120 ∧
        jim_speed * second_meet_time = 300 ∧
        (brenda_speed * first_meet_time + jim_speed * first_meet_time = track_length / 2) ∧
        (brenda_speed * second_meet_time + jim_speed * second_meet_time = track_length) →
        track_length = 280

/-- The circular track length is 280 meters -/
theorem circular_track_length_is_280 : circular_track_length 280 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_circular_track_length_is_280_l188_18800


namespace NUMINAMATH_CALUDE_distance_to_origin_l188_18855

theorem distance_to_origin (a : ℝ) : |a - 0| = 5 → (3 - a = -2 ∨ 3 - a = 8) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l188_18855


namespace NUMINAMATH_CALUDE_statistics_properties_l188_18833

def data : List ℝ := [2, 3, 6, 9, 3, 7]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem statistics_properties :
  mode data = 3 ∧
  median data = 4.5 ∧
  mean data = 5 ∧
  range data = 7 := by sorry

end NUMINAMATH_CALUDE_statistics_properties_l188_18833


namespace NUMINAMATH_CALUDE_evaluate_expression_l188_18866

theorem evaluate_expression : 8^7 + 8^7 + 8^7 - 8^7 = 8^8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l188_18866


namespace NUMINAMATH_CALUDE_circle_radius_when_perimeter_equals_area_l188_18864

/-- Given a square and its circumscribed circle, if the perimeter of the square in inches
    equals the area of the circle in square inches, then the radius of the circle is 8/π inches. -/
theorem circle_radius_when_perimeter_equals_area (s : ℝ) (r : ℝ) :
  s > 0 → r > 0 → s = 2 * r → 4 * s = π * r^2 → r = 8 / π :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_when_perimeter_equals_area_l188_18864


namespace NUMINAMATH_CALUDE_blue_butterflies_count_l188_18810

-- Define the variables
def total_butterflies : ℕ := 11
def black_butterflies : ℕ := 5

-- Define the theorem
theorem blue_butterflies_count :
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black_butterflies = total_butterflies ∧
    blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_butterflies_count_l188_18810


namespace NUMINAMATH_CALUDE_expression_sum_equals_one_l188_18811

theorem expression_sum_equals_one (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hprod : x * y * z = 1) :
  1 / (1 + x + x * y) + y / (1 + y + y * z) + x * z / (1 + z + x * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_sum_equals_one_l188_18811


namespace NUMINAMATH_CALUDE_optimal_room_configuration_l188_18825

/-- Represents a configuration of rooms --/
structure RoomConfiguration where
  large_rooms : ℕ
  small_rooms : ℕ

/-- Checks if a given room configuration is valid for the problem --/
def is_valid_configuration (config : RoomConfiguration) : Prop :=
  3 * config.large_rooms + 2 * config.small_rooms = 26

/-- Calculates the total number of rooms in a configuration --/
def total_rooms (config : RoomConfiguration) : ℕ :=
  config.large_rooms + config.small_rooms

/-- Theorem: The optimal room configuration includes exactly one small room --/
theorem optimal_room_configuration :
  ∃ (config : RoomConfiguration),
    is_valid_configuration config ∧
    (∀ (other : RoomConfiguration), is_valid_configuration other →
      total_rooms config ≤ total_rooms other) ∧
    config.small_rooms = 1 :=
sorry

end NUMINAMATH_CALUDE_optimal_room_configuration_l188_18825


namespace NUMINAMATH_CALUDE_value_after_two_years_approximation_l188_18850

/-- Calculates the value after n years given an initial value and annual increase rate -/
def value_after_n_years (initial_value : ℝ) (increase_rate : ℝ) (n : ℕ) : ℝ :=
  initial_value * (1 + increase_rate) ^ n

/-- The problem statement -/
theorem value_after_two_years_approximation :
  let initial_value : ℝ := 64000
  let increase_rate : ℝ := 1 / 9
  let years : ℕ := 2
  let final_value := value_after_n_years initial_value increase_rate years
  abs (final_value - 79012.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_two_years_approximation_l188_18850


namespace NUMINAMATH_CALUDE_science_class_end_time_l188_18812

-- Define the schedule as a list of durations in minutes
def class_schedule : List ℕ := [60, 90, 25, 45, 15, 75]

-- Function to calculate the end time given a start time and a list of durations
def calculate_end_time (start_time : ℕ) (schedule : List ℕ) : ℕ :=
  start_time + schedule.sum

-- Theorem statement
theorem science_class_end_time :
  calculate_end_time 720 class_schedule = 1030 := by
  sorry

-- Note: 720 minutes is 12:00 pm, 1030 minutes is 5:10 pm

end NUMINAMATH_CALUDE_science_class_end_time_l188_18812


namespace NUMINAMATH_CALUDE_worker_payment_l188_18895

/-- Given a sum of money that can pay worker A for 18 days, worker B for 12 days, 
    and worker C for 24 days, prove that it can pay all three workers together for 5 days. -/
theorem worker_payment (S : ℚ) (A B C : ℚ) (hA : S = 18 * A) (hB : S = 12 * B) (hC : S = 24 * C) :
  ∃ D : ℕ, D = 5 ∧ S = D * (A + B + C) :=
sorry

end NUMINAMATH_CALUDE_worker_payment_l188_18895


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l188_18814

/-- An isosceles right triangle with an inscribed circle -/
structure IsoscelesRightTriangle where
  -- The length of a leg of the triangle
  leg : ℝ
  -- The center of the inscribed circle
  center : ℝ × ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- The area of the inscribed circle is 9π
  circle_area : radius^2 * Real.pi = 9 * Real.pi

/-- The area of an isosceles right triangle with an inscribed circle of area 9π is 36 -/
theorem isosceles_right_triangle_area 
  (triangle : IsoscelesRightTriangle) : triangle.leg^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l188_18814


namespace NUMINAMATH_CALUDE_sum_60_is_negative_120_l188_18890

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_20 : (20 : ℚ) / 2 * (2 * a + 19 * d) = 200
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 50

/-- The sum of the first 60 terms of the arithmetic progression is -120 -/
theorem sum_60_is_negative_120 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -120 := by
  sorry

end NUMINAMATH_CALUDE_sum_60_is_negative_120_l188_18890


namespace NUMINAMATH_CALUDE_curve_single_intersection_l188_18861

/-- The curve (x+2y+a)(x^2-y^2)=0 intersects at a single point if and only if a = 0 -/
theorem curve_single_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, (p.1 + 2 * p.2 + a) * (p.1^2 - p.2^2) = 0) ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_single_intersection_l188_18861


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l188_18822

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 52 = 0 ∧ Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l188_18822


namespace NUMINAMATH_CALUDE_sequence_convergence_comparison_l188_18854

theorem sequence_convergence_comparison
  (k : ℝ) (h_k : 0 < k ∧ k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (h_b : ∀ n, b (n + 1) = (b n) ^ k) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_comparison_l188_18854


namespace NUMINAMATH_CALUDE_geometric_series_sum_l188_18865

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_sum a r n = 7/32 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l188_18865


namespace NUMINAMATH_CALUDE_pizza_slices_left_l188_18863

theorem pizza_slices_left (total_slices : ℕ) (john_slices : ℕ) (sam_multiplier : ℕ) : 
  total_slices = 12 →
  john_slices = 3 →
  sam_multiplier = 2 →
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l188_18863


namespace NUMINAMATH_CALUDE_range_of_T_l188_18830

-- Define the function T
def T (x : ℝ) : ℝ := |2 * x - 1|

-- State the theorem
theorem range_of_T (x : ℝ) : 
  (∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) → 
  x ∈ Set.Ici 2 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_range_of_T_l188_18830


namespace NUMINAMATH_CALUDE_vasechkin_result_l188_18848

def petrov_operation (x : ℚ) : ℚ := (x / 2) * 7 - 1001

def vasechkin_operation (x : ℚ) : ℚ := (x / 8)^2 - 1001

theorem vasechkin_result :
  ∃ x : ℚ, (∃ p : ℕ, Nat.Prime p ∧ petrov_operation x = ↑p) →
  vasechkin_operation x = 295 :=
sorry

end NUMINAMATH_CALUDE_vasechkin_result_l188_18848


namespace NUMINAMATH_CALUDE_cookout_buns_needed_l188_18804

/-- Calculates the number of packs of buns needed for a cookout --/
def buns_needed (total_guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let guests_eating_burgers := total_guests - no_meat_guests
  let total_burgers := guests_eating_burgers * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns_needed :
  buns_needed 10 3 1 1 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_needed_l188_18804


namespace NUMINAMATH_CALUDE_circle_equation_l188_18808

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ ∃ t : ℝ, p.2 = 1 + t}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}

-- Define the center of the circle
def center : ℝ × ℝ := ((-1 : ℝ), (0 : ℝ))

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2}

theorem circle_equation (h1 : center ∈ L ∩ x_axis) 
  (h2 : ∀ p ∈ C, p ∈ tangent_line → (∃ q ∈ C, q ≠ p ∧ Set.Subset C {r | r = p ∨ r = q})) :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 2} := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l188_18808


namespace NUMINAMATH_CALUDE_max_value_implies_m_l188_18844

-- Define the function f(x) = x^2 - 2x + m
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem max_value_implies_m (m : ℝ) :
  (∀ x ∈ interval, f x m ≤ 1) ∧
  (∃ x ∈ interval, f x m = 1) →
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l188_18844


namespace NUMINAMATH_CALUDE_range_of_a_l188_18802

-- Define proposition p
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 
  0 ≤ a ∧ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l188_18802


namespace NUMINAMATH_CALUDE_discriminant_nonnegative_m_value_when_root_difference_is_two_l188_18874

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 4*m*x + 3*m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-4*m)^2 - 4*1*(3*m^2)

-- Theorem 1: The discriminant is always non-negative
theorem discriminant_nonnegative (m : ℝ) : discriminant m ≥ 0 := by
  sorry

-- Theorem 2: When m > 0 and the difference between roots is 2, m = 1
theorem m_value_when_root_difference_is_two (m : ℝ) 
  (h1 : m > 0) 
  (h2 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
                     quadratic_equation m x1 = 0 ∧ 
                     quadratic_equation m x2 = 0 ∧ 
                     x1 - x2 = 2) : 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegative_m_value_when_root_difference_is_two_l188_18874


namespace NUMINAMATH_CALUDE_salt_calculation_l188_18884

/-- Calculates the amount of salt Jack will have after water evaporation -/
def salt_after_evaporation (
  water_volume_day1 : ℝ)
  (water_volume_day2 : ℝ)
  (salt_concentration_day1 : ℝ)
  (salt_concentration_day2 : ℝ)
  (evaporation_rate_day1 : ℝ)
  (evaporation_rate_day2 : ℝ) : ℝ :=
  ((water_volume_day1 * salt_concentration_day1 +
    water_volume_day2 * salt_concentration_day2) * 1000)

theorem salt_calculation :
  salt_after_evaporation 4 4 0.18 0.22 0.30 0.40 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_salt_calculation_l188_18884


namespace NUMINAMATH_CALUDE_expression_simplification_l188_18882

theorem expression_simplification : (((3 + 6 + 9 + 12) / 3) + ((3 * 4 - 6) / 2)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l188_18882


namespace NUMINAMATH_CALUDE_blue_balls_removed_l188_18826

theorem blue_balls_removed (initial_total : Nat) (initial_blue : Nat) (final_probability : Rat) :
  initial_total = 25 →
  initial_blue = 9 →
  final_probability = 1/5 →
  ∃ (removed : Nat), 
    removed ≤ initial_blue ∧
    (initial_blue - removed : Rat) / (initial_total - removed : Rat) = final_probability ∧
    removed = 5 :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l188_18826


namespace NUMINAMATH_CALUDE_segment_properties_l188_18837

/-- Given two points A(1, 2) and B(9, 14), prove the distance between them and their midpoint. -/
theorem segment_properties : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (9, 14)
  let distance := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (distance = 16) ∧ (midpoint = (5, 8)) := by
  sorry

end NUMINAMATH_CALUDE_segment_properties_l188_18837


namespace NUMINAMATH_CALUDE_expression_simplification_l188_18868

theorem expression_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (7 * y^2) * (1 / (2*y)^3) = 17.5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l188_18868


namespace NUMINAMATH_CALUDE_eulers_formula_l188_18888

/-- Euler's formula -/
theorem eulers_formula (a b : ℝ) :
  Complex.exp (a + Complex.I * b) = Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b) := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l188_18888


namespace NUMINAMATH_CALUDE_inequality_not_hold_l188_18894

theorem inequality_not_hold (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1) :
  ¬(a * b < b^2 ∧ b^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_hold_l188_18894


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l188_18831

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 48156 ≡ n [ZMOD 17] ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l188_18831


namespace NUMINAMATH_CALUDE_function_decomposition_l188_18858

open Function Real

theorem function_decomposition (f : ℝ → ℝ) : 
  ∃ (g h : ℝ → ℝ), 
    (∀ x, g (-x) = g x) ∧ 
    (∀ x, h (-x) = -h x) ∧ 
    (∀ x, f x = g x + h x) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l188_18858


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_16_adults_l188_18827

-- Define the problem parameters
def total_cans : ℕ := 8
def adults_per_can : ℕ := 4
def children_per_can : ℕ := 6
def children_fed : ℕ := 24

-- Theorem statement
theorem remaining_soup_feeds_16_adults :
  ∃ (cans_for_children : ℕ) (remaining_cans : ℕ),
    cans_for_children * children_per_can = children_fed ∧
    remaining_cans = total_cans - cans_for_children ∧
    remaining_cans * adults_per_can = 16 :=
by sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_16_adults_l188_18827


namespace NUMINAMATH_CALUDE_max_hubs_is_six_l188_18859

/-- A structure representing a state with cities and roads --/
structure State where
  num_cities : ℕ
  num_roads : ℕ
  num_hubs : ℕ

/-- Definition of a valid state configuration --/
def is_valid_state (s : State) : Prop :=
  s.num_cities = 10 ∧
  s.num_roads = 40 ∧
  s.num_hubs ≤ s.num_cities ∧
  s.num_hubs * (s.num_hubs - 1) / 2 + s.num_hubs * (s.num_cities - s.num_hubs) ≤ s.num_roads

/-- Theorem stating that the maximum number of hubs in a valid state is 6 --/
theorem max_hubs_is_six :
  ∀ s : State, is_valid_state s → s.num_hubs ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_hubs_is_six_l188_18859


namespace NUMINAMATH_CALUDE_initial_average_production_l188_18838

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℚ) 
  (h1 : n = 19)
  (h2 : today_production = 90)
  (h3 : new_average = 52) : 
  ∃ A : ℚ, A = 50 ∧ (A * n + today_production) / (n + 1) = new_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_production_l188_18838


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l188_18805

-- Define the polynomials p and q
def p (c d x : ℝ) : ℝ := x^3 + c*x + d
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 360

-- State the theorem
theorem cubic_roots_problem (c d r s : ℝ) : 
  (p c d r = 0 ∧ p c d s = 0 ∧ q c d (r+5) = 0 ∧ q c d (s-4) = 0) → 
  (d = 84 ∨ d = 1260) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l188_18805


namespace NUMINAMATH_CALUDE_relationship_between_sum_and_product_l188_18820

theorem relationship_between_sum_and_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, a > 0 → b > 0 → (a * b > 1 → a + b > 1)) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b > 1 ∧ a * b ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_sum_and_product_l188_18820


namespace NUMINAMATH_CALUDE_square_divisibility_l188_18891

theorem square_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : ∃ k : ℕ, a^2 + b^2 = k * (a * b + 1)) : 
  ∃ n : ℕ, (a^2 + b^2) / (a * b + 1) = n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l188_18891


namespace NUMINAMATH_CALUDE_minimum_dimes_needed_l188_18899

def shoe_cost : ℚ := 45.50
def five_dollar_bills : ℕ := 4
def one_dollar_coins : ℕ := 10
def dime_value : ℚ := 0.10

theorem minimum_dimes_needed (n : ℕ) : 
  (five_dollar_bills * 5 + one_dollar_coins * 1 + n * dime_value ≥ shoe_cost) →
  n ≥ 155 := by
  sorry

end NUMINAMATH_CALUDE_minimum_dimes_needed_l188_18899


namespace NUMINAMATH_CALUDE_S_n_perfect_square_iff_T_n_perfect_square_iff_l188_18892

/-- Definition of S_n -/
def S_n (n : ℕ) : ℕ := n * (4 * n + 5)

/-- Definition of T_n -/
def T_n (n : ℕ) : ℕ := n * (3 * n + 2)

/-- Definition of is_perfect_square -/
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2

/-- Pell's equation solution -/
def is_pell_solution (l m : ℕ) : Prop := l^2 - 3 * m^2 = 1

/-- Theorem for S_n -/
theorem S_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (S_n n) ↔ n = 1 :=
sorry

/-- Theorem for T_n -/
theorem T_n_perfect_square_iff (n : ℕ) : 
  is_perfect_square (T_n n) ↔ ∃ m : ℕ, n = 2 * m^2 ∧ ∃ l : ℕ, is_pell_solution l m :=
sorry

end NUMINAMATH_CALUDE_S_n_perfect_square_iff_T_n_perfect_square_iff_l188_18892


namespace NUMINAMATH_CALUDE_prime_factors_of_N_l188_18840

def N : ℕ := (10^2011 - 1) / 9

theorem prime_factors_of_N (p : ℕ) (hp : p.Prime) (hdiv : p ∣ N) :
  ∃ j : ℕ, p = 4022 * j + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_N_l188_18840


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l188_18896

theorem triangle_angle_measure (a b c : ℝ) (A C : ℝ) (h : b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C) :
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l188_18896


namespace NUMINAMATH_CALUDE_ellas_raise_percentage_l188_18881

/-- Calculates the percentage raise given the conditions of Ella's babysitting earnings and expenses. -/
theorem ellas_raise_percentage 
  (video_game_percentage : Real) 
  (last_year_video_game_expense : Real) 
  (new_salary : Real) 
  (h1 : video_game_percentage = 0.40)
  (h2 : last_year_video_game_expense = 100)
  (h3 : new_salary = 275) : 
  (new_salary - (last_year_video_game_expense / video_game_percentage)) / (last_year_video_game_expense / video_game_percentage) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellas_raise_percentage_l188_18881


namespace NUMINAMATH_CALUDE_simplify_polynomial_l188_18887

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 9*r - 3) = r^2 - 4*r - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l188_18887


namespace NUMINAMATH_CALUDE_keith_missed_four_games_l188_18879

/-- The number of football games Keith missed, given the total number of games and the number of games he attended. -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Keith missed 4 football games. -/
theorem keith_missed_four_games :
  let total_games : ℕ := 8
  let attended_games : ℕ := 4
  games_missed total_games attended_games = 4 := by
sorry

end NUMINAMATH_CALUDE_keith_missed_four_games_l188_18879


namespace NUMINAMATH_CALUDE_angle_cosine_equality_l188_18835

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side on the ray 3x + 4y = 0 (x ≤ 0), prove that cos(2α + π/6) = (7√3 + 24) / 50 -/
theorem angle_cosine_equality (α : Real) 
    (h1 : ∃ (x y : Real), x ≤ 0 ∧ 3 * x + 4 * y = 0 ∧ 
          x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
          y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
    Real.cos (2 * α + π / 6) = (7 * Real.sqrt 3 + 24) / 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_equality_l188_18835
