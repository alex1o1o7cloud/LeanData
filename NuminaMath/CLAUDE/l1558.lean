import Mathlib

namespace NUMINAMATH_CALUDE_prob_both_genders_selected_l1558_155884

def total_students : ℕ := 8
def male_students : ℕ := 5
def female_students : ℕ := 3
def students_to_select : ℕ := 5

theorem prob_both_genders_selected :
  (Nat.choose total_students students_to_select - Nat.choose male_students students_to_select) /
  Nat.choose total_students students_to_select = 55 / 56 :=
by sorry

end NUMINAMATH_CALUDE_prob_both_genders_selected_l1558_155884


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l1558_155864

theorem rabbit_carrot_problem (rabbit_holes fox_holes : ℕ) : 
  rabbit_holes * 3 = fox_holes * 5 →
  fox_holes = rabbit_holes - 6 →
  rabbit_holes * 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l1558_155864


namespace NUMINAMATH_CALUDE_prove_b_equals_one_l1558_155853

theorem prove_b_equals_one (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 49 * 45 * b) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_equals_one_l1558_155853


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l1558_155863

/-- Theorem: Distance between a point on a parabola and its focus
For a parabola defined by y^2 = 3x, if a point M on the parabola is at a distance
of 1 from the y-axis, then the distance between point M and the focus of the
parabola is 7/4. -/
theorem parabola_point_focus_distance
  (M : ℝ × ℝ) -- Point M on the parabola
  (h_on_parabola : M.2^2 = 3 * M.1) -- M is on the parabola y^2 = 3x
  (h_distance_from_y_axis : M.1 = 1) -- M is at distance 1 from y-axis
  : ∃ F : ℝ × ℝ, -- There exists a focus F
    (F.1 = 3/4 ∧ F.2 = 0) ∧ -- The focus is at (3/4, 0)
    Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 7/4 -- Distance between M and F is 7/4
  := by sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l1558_155863


namespace NUMINAMATH_CALUDE_fraction_simplification_l1558_155843

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 72) = (5 * Real.sqrt 2) / 34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1558_155843


namespace NUMINAMATH_CALUDE_age_sum_is_75_l1558_155882

/-- Given the ages of Alice, Bob, and Carol satisfying certain conditions, prove that the sum of their current ages is 75 years. -/
theorem age_sum_is_75 (alice bob carol : ℕ) : 
  (alice - 10 = (bob - 10) / 2) →  -- 10 years ago, Alice was half of Bob's age
  (4 * alice = 3 * bob) →          -- The ratio of their present ages is 3:4
  (carol = alice + bob + 5) →      -- Carol is 5 years older than the sum of Alice and Bob's current ages
  alice + bob + carol = 75 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_is_75_l1558_155882


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1558_155846

theorem min_sum_of_squares (x y : ℝ) (h : x + y = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (a b : ℝ), a + b = 2 → x^2 + y^2 ≤ a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1558_155846


namespace NUMINAMATH_CALUDE_cube_root_inequality_l1558_155885

theorem cube_root_inequality (a b : ℝ) (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l1558_155885


namespace NUMINAMATH_CALUDE_factorial_sum_perfect_power_l1558_155875

def factorial_sum (n : ℕ+) : ℕ := (Finset.range n).sum (λ i => Nat.factorial (i + 1))

def is_perfect_power (m : ℕ) : Prop := ∃ (a b : ℕ), b > 1 ∧ a^b = m

theorem factorial_sum_perfect_power (n : ℕ+) :
  is_perfect_power (factorial_sum n) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_factorial_sum_perfect_power_l1558_155875


namespace NUMINAMATH_CALUDE_cone_surface_area_l1558_155831

/-- A cone with base radius 1 and lateral surface that unfolds into a semicircle has a total surface area of 3π. -/
theorem cone_surface_area (cone : Real → Real → Real) 
  (h1 : cone 1 2 = 2 * Real.pi) -- Lateral surface area
  (h2 : cone 0 1 = Real.pi) -- Base area
  : cone 0 1 + cone 1 2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1558_155831


namespace NUMINAMATH_CALUDE_equilateral_triangles_area_sum_l1558_155889

/-- Given an isosceles right triangle with leg length 36 units, the sum of the areas
    of an infinite series of equilateral triangles drawn on one leg (with their third
    vertices on the hypotenuse) is equal to half the area of the original right triangle. -/
theorem equilateral_triangles_area_sum (leg_length : ℝ) (h : leg_length = 36) :
  let right_triangle_area := (1 / 2) * leg_length * leg_length
  let equilateral_triangles_area_sum := right_triangle_area / 2
  equilateral_triangles_area_sum = 324 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_area_sum_l1558_155889


namespace NUMINAMATH_CALUDE_sum_product_remainder_l1558_155801

theorem sum_product_remainder : (1789 * 1861 * 1945 + 1533 * 1607 * 1688) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_remainder_l1558_155801


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1558_155888

-- Define the original dimensions
def original_length : ℝ := 140
def original_width : ℝ := 40

-- Define the width decrease percentage
def width_decrease_percent : ℝ := 17.692307692307693

-- Define the expected length increase percentage
def expected_length_increase_percent : ℝ := 21.428571428571427

-- Theorem statement
theorem rectangle_dimension_change :
  let new_width : ℝ := original_width * (1 - width_decrease_percent / 100)
  let new_length : ℝ := (original_length * original_width) / new_width
  let actual_length_increase_percent : ℝ := (new_length - original_length) / original_length * 100
  actual_length_increase_percent = expected_length_increase_percent := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1558_155888


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1558_155814

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem 1
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Icc 0 1 :=
sorry

-- Theorem 2
theorem union_equals_B_iff_a_in_range (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Iic (-2) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_union_equals_B_iff_a_in_range_l1558_155814


namespace NUMINAMATH_CALUDE_height_correction_percentage_l1558_155841

/-- Proves that given a candidate's actual height of 5 feet 8 inches (68 inches),
    and an initial overstatement of 25%, the percentage correction from the
    stated height to the actual height is 20%. -/
theorem height_correction_percentage (actual_height : ℝ) (stated_height : ℝ) :
  actual_height = 68 →
  stated_height = actual_height * 1.25 →
  (stated_height - actual_height) / stated_height * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_height_correction_percentage_l1558_155841


namespace NUMINAMATH_CALUDE_area_difference_S_R_l1558_155804

/-- A square with side length 2 -/
def square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- An isosceles right triangle with legs of length 2 -/
def isoscelesRightTriangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 + p.2 ≤ 2}

/-- Region R: union of the square and 12 isosceles right triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- Region S: smallest convex polygon containing R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_difference_S_R : area S - area R = 36 := by sorry

end NUMINAMATH_CALUDE_area_difference_S_R_l1558_155804


namespace NUMINAMATH_CALUDE_chemical_reaction_result_l1558_155810

-- Define the initial amounts
def initial_silver_nitrate : ℝ := 2
def initial_sodium_hydroxide : ℝ := 2
def initial_hydrochloric_acid : ℝ := 0.5

-- Define the reactions
def main_reaction (x : ℝ) : ℝ := x
def side_reaction (x : ℝ) : ℝ := x

-- Theorem statement
theorem chemical_reaction_result :
  let sodium_hydroxide_in_side_reaction := min initial_sodium_hydroxide initial_hydrochloric_acid
  let remaining_sodium_hydroxide := initial_sodium_hydroxide - sodium_hydroxide_in_side_reaction
  let reaction_limit := min remaining_sodium_hydroxide initial_silver_nitrate
  let sodium_nitrate_formed := main_reaction reaction_limit
  let silver_chloride_formed := main_reaction reaction_limit
  let unreacted_sodium_hydroxide := remaining_sodium_hydroxide - reaction_limit
  sodium_nitrate_formed = 1.5 ∧ 
  silver_chloride_formed = 1.5 ∧ 
  unreacted_sodium_hydroxide = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_chemical_reaction_result_l1558_155810


namespace NUMINAMATH_CALUDE_divisible_by_eleven_smallest_n_seven_l1558_155894

theorem divisible_by_eleven_smallest_n_seven (x : ℕ) : 
  (∃ k : ℕ, x = 11 * k) ∧ 
  (∀ m : ℕ, m < 7 → ¬(∃ j : ℕ, m * 11 = x)) ∧
  (∃ i : ℕ, 7 * 11 = x) →
  x = 77 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_smallest_n_seven_l1558_155894


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l1558_155868

/-- Given an initial number of fish and a number of fish added to an aquarium,
    the total number of fish is equal to the sum of the initial number and the number added. -/
theorem aquarium_fish_count (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l1558_155868


namespace NUMINAMATH_CALUDE_sun_overhead_locations_sun_angle_locations_l1558_155839

/-- Represents a location on Earth by its latitude and longitude -/
structure Location :=
  (lat : Real)
  (lon : Real)

/-- Budapest's location -/
def budapest : Location := ⟨47.5, 19.1⟩

/-- Calculates the location where the Sun is directly overhead given the latitude -/
def overheadLocation (lat : Real) : Location × Location :=
  sorry

/-- Calculates the location where the Sun's rays hit Budapest at a given angle -/
def angleLocation (angle : Real) : Location × Location :=
  sorry

theorem sun_overhead_locations :
  (overheadLocation (-23.5) = (⟨-23.5, 80.8⟩, ⟨-23.5, -42.6⟩)) ∧
  (overheadLocation 0 = (⟨0, 109.1⟩, ⟨0, -70.9⟩)) ∧
  (overheadLocation 23.5 = (⟨23.5, 137.4⟩, ⟨23.5, 99.2⟩)) :=
sorry

theorem sun_angle_locations :
  (angleLocation 60 = (⟨17.5, 129.2⟩, ⟨17.5, -91.0⟩)) ∧
  (angleLocation 30 = (⟨-12.5, 95.1⟩, ⟨-12.5, -56.9⟩)) :=
sorry

end NUMINAMATH_CALUDE_sun_overhead_locations_sun_angle_locations_l1558_155839


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1558_155862

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1558_155862


namespace NUMINAMATH_CALUDE_convex_quadrilateral_area_is_120_l1558_155877

def convex_quadrilateral_area (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧  -- areas are positive
  a < d ∧ b < d ∧ c < d ∧          -- fourth triangle has largest area
  a = 10 ∧ b = 20 ∧ c = 30 →       -- given areas
  a + b + c + d = 120              -- total area

theorem convex_quadrilateral_area_is_120 :
  ∀ a b c d : ℝ, convex_quadrilateral_area a b c d :=
by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_area_is_120_l1558_155877


namespace NUMINAMATH_CALUDE_profit_conditions_l1558_155898

/-- Represents the profit function given the price increase -/
def profit_function (x : ℝ) : ℝ := (50 - 40 + x) * (500 - 10 * x)

/-- Represents the selling price given the price increase -/
def selling_price (x : ℝ) : ℝ := x + 50

/-- Represents the number of units sold given the price increase -/
def units_sold (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating the conditions for achieving a profit of 8000 yuan -/
theorem profit_conditions :
  (∃ x : ℝ, profit_function x = 8000 ∧
    ((selling_price x = 60 ∧ units_sold x = 400) ∨
     (selling_price x = 80 ∧ units_sold x = 200))) :=
by sorry

end NUMINAMATH_CALUDE_profit_conditions_l1558_155898


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1558_155896

theorem divisible_by_nine (k : ℕ+) : 
  ∃ n : ℤ, 3 * (2 + 7^(k.val)) = 9 * n := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1558_155896


namespace NUMINAMATH_CALUDE_solve_for_c_l1558_155866

theorem solve_for_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y) / 20 + (c * y) / 10 = 0.6 * y) : c = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l1558_155866


namespace NUMINAMATH_CALUDE_total_keys_for_tim_l1558_155883

/-- Calculates the total number of keys needed for Tim's rental properties -/
def total_keys (apartment_complex_1 apartment_complex_2 apartment_complex_3 : ℕ)
  (individual_houses : ℕ)
  (keys_per_apartment keys_per_main_entrance keys_per_house : ℕ) : ℕ :=
  (apartment_complex_1 + apartment_complex_2 + apartment_complex_3) * keys_per_apartment +
  3 * keys_per_main_entrance +
  individual_houses * keys_per_house

/-- Theorem stating the total number of keys needed for Tim's rental properties -/
theorem total_keys_for_tim : 
  total_keys 16 20 24 4 4 10 6 = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_keys_for_tim_l1558_155883


namespace NUMINAMATH_CALUDE_even_function_solution_set_l1558_155800

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f x < 0}

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_zero : f (-4) = 0 ∧ f 2 = 0)
  (h_decreasing : ∀ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, x < y → f x > f y)
  (h_increasing : ∀ x ∈ Set.Ici 3, ∀ y ∈ Set.Ici 3, x < y → f x < f y) :
  solution_set f = Set.union (Set.union (Set.Iio (-4)) (Set.Ioo (-2) 0)) (Set.Ioo 2 4) :=
sorry

end NUMINAMATH_CALUDE_even_function_solution_set_l1558_155800


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l1558_155856

theorem cube_sum_implies_sum_bound (a b : ℝ) :
  a > 0 → b > 0 → a^3 + b^3 = 2 → a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l1558_155856


namespace NUMINAMATH_CALUDE_product_of_fractions_l1558_155834

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1558_155834


namespace NUMINAMATH_CALUDE_contest_probability_l1558_155890

/-- The probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- The number of preset questions -/
def n : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := 2 * p^3 * (1 - p)

theorem contest_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_contest_probability_l1558_155890


namespace NUMINAMATH_CALUDE_room_painting_cost_l1558_155832

/-- Calculate the cost of painting a room's walls given its dimensions and openings. -/
def paintingCost (roomLength roomWidth roomHeight : ℝ)
                 (doorCount doorLength doorHeight : ℝ)
                 (largeWindowCount largeWindowLength largeWindowHeight : ℝ)
                 (smallWindowCount smallWindowLength smallWindowHeight : ℝ)
                 (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := doorCount * doorLength * doorHeight
  let largeWindowArea := largeWindowCount * largeWindowLength * largeWindowHeight
  let smallWindowArea := smallWindowCount * smallWindowLength * smallWindowHeight
  let paintableArea := wallArea - (doorArea + largeWindowArea + smallWindowArea)
  paintableArea * costPerSqm

/-- Theorem stating that the cost of painting the room with given dimensions is 474 Rs. -/
theorem room_painting_cost :
  paintingCost 10 7 5 2 1 3 1 2 1.5 2 1 1.5 3 = 474 := by
  sorry

end NUMINAMATH_CALUDE_room_painting_cost_l1558_155832


namespace NUMINAMATH_CALUDE_octagon_diagonals_l1558_155872

/-- The number of diagonals in an octagon -/
def diagonals_in_octagon : ℕ :=
  let vertices : ℕ := 8
  let sides : ℕ := 8
  (vertices.choose 2) - sides

/-- Theorem stating that the number of diagonals in an octagon is 20 -/
theorem octagon_diagonals :
  diagonals_in_octagon = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l1558_155872


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1558_155825

theorem arithmetic_expression_equality : 3^2 + 4 * 2 - 6 / 3 + 7 = 22 := by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1558_155825


namespace NUMINAMATH_CALUDE_triangle_properties_l1558_155861

theorem triangle_properties (a b c : ℝ) (A B C : Real) (S : ℝ) (D : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  a * Real.sin B = b * Real.sin (A + π / 3) →
  S = 2 * Real.sqrt 3 →
  S = (1 / 2) * b * c * Real.sin A →
  D.1 = (2 / 3) * b →
  D.2 = 0 →
  (∃ (AD : ℝ), AD ≥ (4 * Real.sqrt 3) / 3 ∧
    AD^2 = (1 / 9) * c^2 + (4 / 9) * b^2 + (16 / 9)) →
  A = π / 3 ∧ (∃ (AD_min : ℝ), AD_min = (4 * Real.sqrt 3) / 3 ∧
    ∀ (AD : ℝ), AD ≥ AD_min) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1558_155861


namespace NUMINAMATH_CALUDE_expression_factorization_l1558_155878

theorem expression_factorization (x : ℝ) : 
  (20 * x^3 - 100 * x^2 + 90 * x - 10) - (5 * x^3 - 10 * x^2 + 5) = 
  15 * (x^3 - 6 * x^2 + 6 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1558_155878


namespace NUMINAMATH_CALUDE_simplify_fraction_l1558_155893

theorem simplify_fraction :
  ∀ x : ℝ, x > 0 → 
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  Real.sqrt ((297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1558_155893


namespace NUMINAMATH_CALUDE_digit_sum_properties_l1558_155886

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if two natural numbers have the same digits in a different order -/
def same_digits (m k : ℕ) : Prop := sorry

theorem digit_sum_properties (M K : ℕ) (h : same_digits M K) :
  (sum_of_digits (2 * M) = sum_of_digits (2 * K)) ∧
  (M % 2 = 0 → K % 2 = 0 → sum_of_digits (M / 2) = sum_of_digits (K / 2)) ∧
  (sum_of_digits (5 * M) = sum_of_digits (5 * K)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_properties_l1558_155886


namespace NUMINAMATH_CALUDE_problem_statement_l1558_155817

theorem problem_statement (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 2) : 
  x^4 + c*d*x^2 - a - b = 20 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1558_155817


namespace NUMINAMATH_CALUDE_cube_sum_greater_than_product_sum_l1558_155829

theorem cube_sum_greater_than_product_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_greater_than_product_sum_l1558_155829


namespace NUMINAMATH_CALUDE_mrs_hilt_saw_twelve_legs_l1558_155855

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Mrs. Hilt saw -/
def dogs_seen : ℕ := 2

/-- The number of chickens Mrs. Hilt saw -/
def chickens_seen : ℕ := 2

/-- The total number of animal legs Mrs. Hilt saw -/
def total_legs : ℕ := dogs_seen * dog_legs + chickens_seen * chicken_legs

theorem mrs_hilt_saw_twelve_legs : total_legs = 12 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_saw_twelve_legs_l1558_155855


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l1558_155821

/-- The measure of one interior angle of a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ :=
  let n : ℕ := 6  -- number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)
  let num_angles : ℕ := n
  sum_interior_angles / num_angles

/-- The result of regular_hexagon_interior_angle is equal to 120. -/
theorem regular_hexagon_interior_angle_is_120 : 
  regular_hexagon_interior_angle = 120 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l1558_155821


namespace NUMINAMATH_CALUDE_father_child_ages_l1558_155858

theorem father_child_ages : ∃ (f b : ℕ), 
  13 ≤ b ∧ b ≤ 19 ∧ 
  100 * f + b - (f - b) = 4289 ∧ 
  f + b = 59 := by
sorry

end NUMINAMATH_CALUDE_father_child_ages_l1558_155858


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1558_155813

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (h1 : a 2 = 1)  -- given: a2 = 1
  (h2 : a 6 = 13)  -- given: a6 = 13
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1558_155813


namespace NUMINAMATH_CALUDE_well_diameter_l1558_155808

/-- The diameter of a circular well given its depth and volume -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (h1 : depth = 14) (h2 : volume = 43.982297150257104) :
  let radius := Real.sqrt (volume / (Real.pi * depth))
  2 * radius = 2 := by sorry

end NUMINAMATH_CALUDE_well_diameter_l1558_155808


namespace NUMINAMATH_CALUDE_f_extrema_l1558_155811

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + 1

theorem f_extrema :
  let I : Set ℝ := Set.Icc 0 (Real.pi / 2)
  (∀ x ∈ I, f x ≥ 1) ∧
  (∀ x ∈ I, f x ≤ 3 + Real.sqrt 3) ∧
  (∃ x ∈ I, f x = 1) ∧
  (∃ x ∈ I, f x = 3 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l1558_155811


namespace NUMINAMATH_CALUDE_cistern_fill_time_theorem_l1558_155833

/-- Represents the rate at which a pipe can fill or empty a cistern -/
structure PipeRate where
  fill : ℚ  -- Fraction of cistern filled or emptied
  time : ℚ  -- Time taken in minutes
  deriving Repr

/-- Calculates the rate of filling or emptying per minute -/
def rate_per_minute (p : PipeRate) : ℚ := p.fill / p.time

/-- Represents the problem of filling a cistern with multiple pipes -/
structure CisternProblem where
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_c : PipeRate
  target_fill : ℚ
  deriving Repr

/-- Calculates the time required to fill the target amount of the cistern -/
def fill_time (problem : CisternProblem) : ℚ :=
  let combined_rate := rate_per_minute problem.pipe_a + rate_per_minute problem.pipe_b - rate_per_minute problem.pipe_c
  problem.target_fill / combined_rate

/-- The main theorem stating the time required to fill half the cistern -/
theorem cistern_fill_time_theorem (problem : CisternProblem) 
  (h1 : problem.pipe_a = ⟨1/2, 10⟩)
  (h2 : problem.pipe_b = ⟨2/3, 15⟩)
  (h3 : problem.pipe_c = ⟨1/4, 20⟩)
  (h4 : problem.target_fill = 1/2) :
  fill_time problem = 720/118 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_theorem_l1558_155833


namespace NUMINAMATH_CALUDE_tangent_line_sin_plus_one_l1558_155812

/-- The equation of the tangent line to y = sin x + 1 at (0, 1) is x - y + 1 = 0 -/
theorem tangent_line_sin_plus_one (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.sin t + 1
  let df : ℝ → ℝ := λ t => Real.cos t
  let tangent_point : ℝ × ℝ := (0, 1)
  let tangent_slope : ℝ := df tangent_point.1
  x - y + 1 = 0 ↔ y = tangent_slope * (x - tangent_point.1) + tangent_point.2 :=
by
  sorry

#check tangent_line_sin_plus_one

end NUMINAMATH_CALUDE_tangent_line_sin_plus_one_l1558_155812


namespace NUMINAMATH_CALUDE_letter_at_unknown_position_l1558_155828

/-- Represents the letters that can be used in the grid -/
inductive Letter : Type
| A | B | C | D | E

/-- Represents a position in the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid -/
def Grid := Position → Letter

/-- Check if each letter appears exactly once in each row -/
def valid_rows (g : Grid) : Prop :=
  ∀ r : Fin 5, ∀ l : Letter, ∃! c : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in each column -/
def valid_columns (g : Grid) : Prop :=
  ∀ c : Fin 5, ∀ l : Letter, ∃! r : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in the main diagonal -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, i⟩ = l

/-- Check if each letter appears exactly once in the anti-diagonal -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, 4 - i⟩ = l

/-- Check if the grid satisfies all constraints -/
def valid_grid (g : Grid) : Prop :=
  valid_rows g ∧ valid_columns g ∧ valid_main_diagonal g ∧ valid_anti_diagonal g

/-- The theorem to prove -/
theorem letter_at_unknown_position (g : Grid) 
  (h_valid : valid_grid g)
  (h_A : g ⟨0, 0⟩ = Letter.A)
  (h_D : g ⟨3, 0⟩ = Letter.D)
  (h_E : g ⟨4, 0⟩ = Letter.E) :
  ∃ p : Position, g p = Letter.B :=
by sorry

end NUMINAMATH_CALUDE_letter_at_unknown_position_l1558_155828


namespace NUMINAMATH_CALUDE_right_triangle_area_l1558_155805

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTriangle where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x - 2
  median_P : ∃ (t : ℝ), (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t - 2
  -- Median through Q lies along y = 3x + 3
  median_Q : ∃ (t : ℝ), (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 3 * t + 3

/-- The area of the right triangle PQR is 290 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 290 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1558_155805


namespace NUMINAMATH_CALUDE_average_rainfall_proof_l1558_155826

/-- The average rainfall for the first three days of May in a normal year -/
def average_rainfall : ℝ := 140

/-- Rainfall on the first day in cm -/
def first_day_rainfall : ℝ := 26

/-- Rainfall on the second day in cm -/
def second_day_rainfall : ℝ := 34

/-- Rainfall difference between second and third day in cm -/
def third_day_difference : ℝ := 12

/-- Difference between this year's total rainfall and average in cm -/
def rainfall_difference : ℝ := 58

theorem average_rainfall_proof :
  let third_day_rainfall := second_day_rainfall - third_day_difference
  let this_year_total := first_day_rainfall + second_day_rainfall + third_day_rainfall
  average_rainfall = this_year_total + rainfall_difference := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_proof_l1558_155826


namespace NUMINAMATH_CALUDE_correct_calculation_l1558_155842

theorem correct_calculation (x : ℝ) : 2 * x^2 - x^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1558_155842


namespace NUMINAMATH_CALUDE_reciprocal_of_one_l1558_155891

-- Define the concept of reciprocal
def is_reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem statement
theorem reciprocal_of_one : is_reciprocal 1 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_l1558_155891


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1558_155803

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1558_155803


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1558_155879

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3672 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 → (3672 * 10 + M) % 6 = 0 → M ≤ N :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1558_155879


namespace NUMINAMATH_CALUDE_log_continuous_l1558_155824

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_continuous (b : ℝ) (h₁ : b > 0) (h₂ : b ≠ 1) :
  ∀ a : ℝ, a > 0 → ContinuousAt (log b) a :=
by sorry

end NUMINAMATH_CALUDE_log_continuous_l1558_155824


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1558_155870

theorem quadratic_roots_property : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ + 1 = 0 → 
  x₂^2 - 3*x₂ + 1 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + 3*x₂ + x₁*x₂ - 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1558_155870


namespace NUMINAMATH_CALUDE_contradiction_assumption_correct_l1558_155897

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def isObtuse (angle : ℝ) : Prop := angle > 90

/-- The statement "A triangle has at most one obtuse angle". -/
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), (isObtuse a → ¬isObtuse b ∧ ¬isObtuse c) ∧
                 (isObtuse b → ¬isObtuse a ∧ ¬isObtuse c) ∧
                 (isObtuse c → ¬isObtuse a ∧ ¬isObtuse b)

/-- The correct assumption for the method of contradiction. -/
def correctAssumption (t : Triangle) : Prop :=
  ∃ (a b : ℝ), isObtuse a ∧ isObtuse b ∧ a ≠ b

theorem contradiction_assumption_correct (t : Triangle) :
  ¬atMostOneObtuseAngle t ↔ correctAssumption t :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_correct_l1558_155897


namespace NUMINAMATH_CALUDE_prism_18_edges_8_faces_l1558_155851

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_18_edges_8_faces_l1558_155851


namespace NUMINAMATH_CALUDE_total_angles_count_l1558_155840

/-- The number of 90° angles in a rectangle -/
def rectangleAngles : ℕ := 4

/-- The number of 90° angles in a square -/
def squareAngles : ℕ := 4

/-- The number of rectangular flower beds in the park -/
def flowerBeds : ℕ := 3

/-- The number of square goal areas in the football field -/
def goalAreas : ℕ := 4

/-- The total number of 90° angles in the park and football field -/
def totalAngles : ℕ := 
  rectangleAngles + flowerBeds * rectangleAngles + 
  squareAngles + goalAreas * squareAngles

theorem total_angles_count : totalAngles = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_angles_count_l1558_155840


namespace NUMINAMATH_CALUDE_equation_roots_l1558_155860

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  x₁ ≠ x₂ ∧ 
  x₁ = (29 + Real.sqrt 457) / 24 ∧ 
  x₂ = (29 - Real.sqrt 457) / 24 ∧ 
  ∀ x : ℝ, x ≠ 2 → 
    3 * x^2 / (x - 2) - (x + 4) / 4 + (7 - 9 * x) / (x - 2) + 2 = 0 ↔ 
    (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l1558_155860


namespace NUMINAMATH_CALUDE_circular_seating_theorem_l1558_155806

/-- The number of people seated at a circular table. -/
def n : ℕ := sorry

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- The theorem stating that if the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of n people, then n must be 41. -/
theorem circular_seating_theorem :
  circularDistance 31 7 = circularDistance 31 14 → n = 41 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_theorem_l1558_155806


namespace NUMINAMATH_CALUDE_sequence_term_proof_l1558_155835

def sequence_sum (n : ℕ) := 3^n + 2

def sequence_term (n : ℕ) : ℝ :=
  if n = 1 then 5 else 2 * 3^(n-1)

theorem sequence_term_proof (n : ℕ) :
  sequence_term n = 
    if n = 1 
    then sequence_sum 1
    else sequence_sum n - sequence_sum (n-1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l1558_155835


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1558_155899

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The condition that a = 3 is sufficient for the lines to be parallel -/
theorem sufficient_condition (a : ℝ) :
  a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The condition that a = 3 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2) :=
by sorry

/-- The main theorem stating that a = 3 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = 3 → are_parallel 2 a 1 (a - 1) 3 (-2)) ∧
  (∃ a : ℝ, a ≠ 3 ∧ are_parallel 2 a 1 (a - 1) 3 (-2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1558_155899


namespace NUMINAMATH_CALUDE_martian_amoeba_nim_exists_l1558_155845

-- Define the set of Martian amoebas
inductive MartianAmoeba
  | A
  | B
  | C

-- Define the function type
def AmoebaNim := MartianAmoeba → Nat

-- Define the bitwise XOR operation
def bxor (a b : Nat) : Nat :=
  Nat.xor a b

-- State the theorem
theorem martian_amoeba_nim_exists : ∃ (f : AmoebaNim),
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.B) = f MartianAmoeba.C) ∧
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.C) = f MartianAmoeba.B) ∧
  (bxor (f MartianAmoeba.B) (f MartianAmoeba.C) = f MartianAmoeba.A) :=
by
  sorry

end NUMINAMATH_CALUDE_martian_amoeba_nim_exists_l1558_155845


namespace NUMINAMATH_CALUDE_like_terms_power_l1558_155850

/-- 
Given two monomials x^(a+3)y and -5xy^b that are like terms,
prove that (a+b)^2023 = -1
-/
theorem like_terms_power (a b : ℤ) : 
  (a + 3 = 1 ∧ b = 1) → (a + b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_like_terms_power_l1558_155850


namespace NUMINAMATH_CALUDE_puzzle_solution_l1558_155867

theorem puzzle_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 17) : 
  w - y = 229 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1558_155867


namespace NUMINAMATH_CALUDE_unripe_oranges_per_day_is_65_l1558_155822

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 390

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := total_unripe_oranges / harvest_days

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 65 -/
theorem unripe_oranges_per_day_is_65 : unripe_oranges_per_day = 65 := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_per_day_is_65_l1558_155822


namespace NUMINAMATH_CALUDE_orchid_bushes_after_planting_l1558_155859

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: Given 22 initial orchid bushes and 13 newly planted orchid bushes,
    the total number of orchid bushes after planting will be 35. -/
theorem orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_after_planting_l1558_155859


namespace NUMINAMATH_CALUDE_stone_151_is_9_l1558_155865

/-- Represents the number of stones in the arrangement. -/
def num_stones : ℕ := 12

/-- Represents the modulus for the counting pattern. -/
def counting_modulus : ℕ := 22

/-- The number we want to find the original stone for. -/
def target_count : ℕ := 151

/-- Function to determine the original stone number given a count. -/
def original_stone (count : ℕ) : ℕ :=
  (count - 1) % counting_modulus + 1

theorem stone_151_is_9 : original_stone target_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_stone_151_is_9_l1558_155865


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l1558_155818

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_subtraction_proof :
  let base7_num := base_to_decimal [5, 4, 3, 2, 1, 0] 7
  let base8_num := base_to_decimal [4, 5, 3, 2, 1] 8
  base7_num - base8_num = 75620 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l1558_155818


namespace NUMINAMATH_CALUDE_problem_solution_l1558_155849

theorem problem_solution (x : ℝ) (n : ℝ) (h1 : x > 0) 
  (h2 : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1558_155849


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1558_155809

theorem divisibility_by_five (B : ℕ) : 
  B < 10 → (947 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1558_155809


namespace NUMINAMATH_CALUDE_divide_by_three_l1558_155874

theorem divide_by_three (x : ℚ) (h : x / 4 = 12) : x / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_three_l1558_155874


namespace NUMINAMATH_CALUDE_password_20_combinations_l1558_155847

def password_combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem password_20_combinations :
  ∃ (k : ℕ), k ≤ 5 ∧ password_combinations 5 k = 20 ↔ k = 3 :=
sorry

end NUMINAMATH_CALUDE_password_20_combinations_l1558_155847


namespace NUMINAMATH_CALUDE_complex_magnitude_l1558_155895

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_magnitude (a b : ℝ) (h : a / (1 - i) = 1 - b * i) : 
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1558_155895


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1558_155873

theorem quadratic_equation_solutions : ∃ (k : ℝ),
  (∀ (x : ℝ), k * x^2 - 7 * x - 6 = 0 ↔ (x = 2 ∨ x = -3/2)) ∧ k = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1558_155873


namespace NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l1558_155844

/-- Represents a metro line with a specific round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Calculates the least common multiple (LCM) of three natural numbers -/
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

/-- Theorem: All trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (system : MetroSystem) : 
  (2016 % lcm3 system.redLine.roundTripTime system.blueLine.roundTripTime system.greenLine.roundTripTime = 0) → 
  (∀ (line : MetroLine), 2016 % line.roundTripTime = 0) :=
by
  sorry

/-- The actual metro system of city N -/
def cityN : MetroSystem :=
  { redLine := { roundTripTime := 14 }
  , blueLine := { roundTripTime := 16 }
  , greenLine := { roundTripTime := 18 }
  }

/-- Proof that the trains in city N return to their initial positions after 2016 minutes -/
theorem cityN_trains_return_to_initial_positions : 
  (2016 % lcm3 cityN.redLine.roundTripTime cityN.blueLine.roundTripTime cityN.greenLine.roundTripTime = 0) ∧
  (∀ (line : MetroLine), line ∈ [cityN.redLine, cityN.blueLine, cityN.greenLine] → 2016 % line.roundTripTime = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l1558_155844


namespace NUMINAMATH_CALUDE_pig_count_after_joining_l1558_155836

theorem pig_count_after_joining (initial_pigs joining_pigs : ℕ) :
  initial_pigs = 64 →
  joining_pigs = 22 →
  initial_pigs + joining_pigs = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_pig_count_after_joining_l1558_155836


namespace NUMINAMATH_CALUDE_rounding_estimate_l1558_155827

theorem rounding_estimate (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (a' : ℕ) (ha' : a' ≥ a)
  (b' : ℕ) (hb' : b' ≤ b)
  (c' : ℕ) (hc' : c' ≥ c)
  (d' : ℕ) (hd' : d' ≥ d) :
  (a' * d' : ℚ) / b' + c' > (a * d : ℚ) / b + c :=
sorry

end NUMINAMATH_CALUDE_rounding_estimate_l1558_155827


namespace NUMINAMATH_CALUDE_prime_divisor_congruence_l1558_155815

theorem prime_divisor_congruence (p q : ℕ) : 
  Prime p → 
  Prime q → 
  q ∣ ((p^p - 1) / (p - 1)) → 
  q ≡ 1 [ZMOD p] := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_congruence_l1558_155815


namespace NUMINAMATH_CALUDE_most_likely_white_balls_l1558_155848

/-- Represents a box of balls -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  white_le_total : white ≤ total
  black_eq_total_sub_white : black = total - white

/-- Represents the result of multiple draws -/
structure DrawResult where
  total_draws : ℕ
  white_draws : ℕ
  white_draws_le_total : white_draws ≤ total_draws

/-- The probability of drawing a white ball given a box configuration -/
def draw_probability (box : BallBox) : ℚ :=
  box.white / box.total

/-- The likelihood of a draw result given a box configuration -/
def draw_likelihood (box : BallBox) (result : DrawResult) : ℚ :=
  (draw_probability box) ^ result.white_draws * (1 - draw_probability box) ^ (result.total_draws - result.white_draws)

/-- Theorem: Given 10 balls and 240 white draws out of 400, 6 white balls is most likely -/
theorem most_likely_white_balls 
  (box : BallBox) 
  (result : DrawResult) 
  (h_total : box.total = 10) 
  (h_draws : result.total_draws = 400) 
  (h_white_draws : result.white_draws = 240) :
  (∀ (other_box : BallBox), other_box.total = 10 → 
    draw_likelihood box result ≥ draw_likelihood other_box result) →
  box.white = 6 :=
sorry

end NUMINAMATH_CALUDE_most_likely_white_balls_l1558_155848


namespace NUMINAMATH_CALUDE_pablo_share_fraction_l1558_155881

/-- Represents the number of eggs each person has -/
structure EggDistribution :=
  (mia : ℕ)
  (sofia : ℕ)
  (pablo : ℕ)
  (juan : ℕ)

/-- The initial distribution of eggs -/
def initial_distribution (m : ℕ) : EggDistribution :=
  { mia := m
  , sofia := 3 * m
  , pablo := 12 * m
  , juan := 5 }

/-- The fraction of eggs Pablo gives to Sofia -/
def pablo_to_sofia_fraction (m : ℕ) : ℚ :=
  (4 * m + 5 : ℚ) / (48 * m : ℚ)

theorem pablo_share_fraction (m : ℕ) :
  let init := initial_distribution m
  let total := init.mia + init.sofia + init.pablo + init.juan
  let equal_share := total / 4
  let sofia_needs := equal_share - init.sofia
  sofia_needs / init.pablo = pablo_to_sofia_fraction m := by
  sorry

end NUMINAMATH_CALUDE_pablo_share_fraction_l1558_155881


namespace NUMINAMATH_CALUDE_three_solutions_sum_and_m_value_m_range_for_positive_f_l1558_155837

noncomputable section

def f (m : ℝ) (x : ℝ) := 4 - m * Real.sin x - 3 * (Real.cos x)^2

theorem three_solutions_sum_and_m_value 
  (m : ℝ) 
  (h₁ : ∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < π ∧ 
                       0 < x₂ ∧ x₂ < π ∧ 
                       0 < x₃ ∧ x₃ < π ∧ 
                       x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
                       f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m x₃ = 0) : 
  m = 4 ∧ ∃ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 3 * π / 2 :=
sorry

theorem m_range_for_positive_f 
  (m : ℝ) 
  (h : ∀ x : ℝ, -π/6 ≤ x ∧ x ≤ π → f m x > 0) : 
  -7/2 < m ∧ m < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_sum_and_m_value_m_range_for_positive_f_l1558_155837


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1558_155807

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : Int | ∃ k : Int, x = 3 * k} :=
by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1558_155807


namespace NUMINAMATH_CALUDE_robin_cupcakes_l1558_155876

/-- Calculates the total number of cupcakes Robin has after baking and selling. -/
def total_cupcakes (initial : ℕ) (sold : ℕ) (additional : ℕ) : ℕ :=
  initial - sold + additional

/-- Theorem stating that Robin has 59 cupcakes in total. -/
theorem robin_cupcakes : total_cupcakes 42 22 39 = 59 := by
  sorry

end NUMINAMATH_CALUDE_robin_cupcakes_l1558_155876


namespace NUMINAMATH_CALUDE_store_uniforms_l1558_155869

theorem store_uniforms (total_uniforms : ℕ) (additional_uniform : ℕ) : 
  total_uniforms = 927 → 
  additional_uniform = 1 → 
  ∃ (employees : ℕ), 
    employees > 1 ∧ 
    (total_uniforms + additional_uniform) % employees = 0 ∧ 
    total_uniforms % employees ≠ 0 ∧
    ∀ (n : ℕ), n > employees → (total_uniforms + additional_uniform) % n ≠ 0 ∨ total_uniforms % n = 0 →
    employees = 29 := by
sorry

end NUMINAMATH_CALUDE_store_uniforms_l1558_155869


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1558_155854

theorem difference_of_squares_special_case : (733 : ℤ) * 733 - 732 * 734 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1558_155854


namespace NUMINAMATH_CALUDE_intersection_point_l1558_155880

-- Define the system of equations
def system (x y m n : ℝ) : Prop :=
  2 * x + y = m ∧ x - y = n

-- Define the solution to the system
def solution : ℝ × ℝ := (-1, 3)

-- Define the lines
def line1 (x y m : ℝ) : Prop := y = -2 * x + m
def line2 (x y n : ℝ) : Prop := y = x - n

-- Theorem statement
theorem intersection_point :
  ∀ (m n : ℝ),
  system (solution.1) (solution.2) m n →
  ∃ (x y : ℝ), 
    line1 x y m ∧ 
    line2 x y n ∧ 
    x = solution.1 ∧ 
    y = solution.2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1558_155880


namespace NUMINAMATH_CALUDE_bob_winning_strategy_l1558_155857

/-- Represents the state of the game with the number of beads -/
structure GameState where
  beads : Nat
  deriving Repr

/-- Represents a player in the game -/
inductive Player
  | Alice
  | Bob
  deriving Repr

/-- Defines a valid move in the game -/
def validMove (s : GameState) : Prop :=
  s.beads > 1

/-- Defines the next player's turn -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Alice

/-- Theorem stating that Bob has a winning strategy -/
theorem bob_winning_strategy :
  ∀ (initialBeads : Nat),
    initialBeads % 2 = 1 →
    ∃ (strategy : Player → GameState → Nat),
      ∀ (game : GameState),
        game.beads = initialBeads →
        ¬(∃ (aliceStrategy : Player → GameState → Nat),
          ∀ (state : GameState),
            validMove state →
            (state.beads % 2 = 1 → 
              validMove {beads := state.beads - strategy Player.Bob state} ∧
              validMove {beads := strategy Player.Bob state}) ∧
            (state.beads % 2 = 0 →
              validMove {beads := state.beads - aliceStrategy Player.Alice state} ∧
              validMove {beads := aliceStrategy Player.Alice state})) :=
sorry

#check bob_winning_strategy

end NUMINAMATH_CALUDE_bob_winning_strategy_l1558_155857


namespace NUMINAMATH_CALUDE_x_value_is_six_l1558_155820

def star_op (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_is_six (x : ℝ) : star_op 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_is_six_l1558_155820


namespace NUMINAMATH_CALUDE_enrollment_calculation_l1558_155871

def final_enrollment (initial : ℕ) (new_interested : ℕ) (new_dropout_rate : ℚ)
  (additional_dropouts : ℕ) (increase_factor : ℕ) (schedule_dropouts : ℕ)
  (final_rally : ℕ) (later_dropout_rate : ℚ) (graduation_rate : ℚ) : ℕ :=
  sorry

theorem enrollment_calculation :
  final_enrollment 8 8 (1/4) 2 5 2 6 (1/2) (1/2) = 19 :=
sorry

end NUMINAMATH_CALUDE_enrollment_calculation_l1558_155871


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l1558_155830

theorem fraction_to_zero_power :
  let x : ℚ := -123456789 / 9876543210
  x ≠ 0 →
  x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l1558_155830


namespace NUMINAMATH_CALUDE_triangle_side_length_l1558_155852

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 2 →
  c = 2 * Real.sqrt 3 →
  Real.cos A = Real.sqrt 3 / 2 →
  b < c →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1558_155852


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l1558_155816

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive factors -/
def has_twelve_factors (n : ℕ) : Prop := count_factors n = 12

/-- Theorem stating that 108 is the least positive integer with exactly 12 positive factors -/
theorem least_with_twelve_factors :
  (∀ m : ℕ, m > 0 → m < 108 → ¬(has_twelve_factors m)) ∧ has_twelve_factors 108 := by
  sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l1558_155816


namespace NUMINAMATH_CALUDE_max_a_value_l1558_155838

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem max_a_value (t : ℝ) (h : t ∈ Set.Ioo 0 2) :
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a) →
  (∃ a : ℝ, (∀ x ∈ Set.Icc (-1) 2, f t x > x + a) ∧ a = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1558_155838


namespace NUMINAMATH_CALUDE_house_development_problem_l1558_155802

theorem house_development_problem (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85)
  (h2 : garage = 50)
  (h3 : pool = 40)
  (h4 : both = 35) :
  total - (garage + pool - both) = 30 :=
by sorry

end NUMINAMATH_CALUDE_house_development_problem_l1558_155802


namespace NUMINAMATH_CALUDE_remainder_problem_l1558_155823

theorem remainder_problem (a : ℤ) : ∃ (n : ℕ), n > 1 ∧
  (1108 + a) % n = 4 ∧
  1453 % n = 4 ∧
  (1844 + 2*a) % n = 4 ∧
  2281 % n = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1558_155823


namespace NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l1558_155819

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_one_fourth_l1558_155819


namespace NUMINAMATH_CALUDE_paper_piles_theorem_l1558_155892

theorem paper_piles_theorem (n : ℕ) :
  1000 < n ∧ n < 2000 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) →
  ∃ m : ℕ, m = 41 ∧ m ≠ 1 ∧ m ≠ n ∧ n % m = 0 :=
by sorry

end NUMINAMATH_CALUDE_paper_piles_theorem_l1558_155892


namespace NUMINAMATH_CALUDE_smallest_with_ten_divisors_l1558_155887

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 positive divisors -/
def hasTenDivisors (n : ℕ) : Prop := countDivisors n = 10

/-- The theorem stating that 48 is the smallest natural number with exactly 10 positive divisors -/
theorem smallest_with_ten_divisors :
  (∀ m : ℕ, m < 48 → ¬(hasTenDivisors m)) ∧ hasTenDivisors 48 := by sorry

end NUMINAMATH_CALUDE_smallest_with_ten_divisors_l1558_155887
