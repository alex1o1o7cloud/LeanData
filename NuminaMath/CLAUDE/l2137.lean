import Mathlib

namespace NUMINAMATH_CALUDE_function_domain_range_l2137_213778

theorem function_domain_range (a : ℝ) (h1 : a > 1) : 
  (∀ x ∈ Set.Icc 1 a, x^2 - 2*a*x + 5 ∈ Set.Icc 1 a) ∧
  (∀ y ∈ Set.Icc 1 a, ∃ x ∈ Set.Icc 1 a, y = x^2 - 2*a*x + 5) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_l2137_213778


namespace NUMINAMATH_CALUDE_fraction_of_y_l2137_213769

theorem fraction_of_y (w x y : ℝ) (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 / w + 2 / x = 2 / y) → (w * x = y) → ((w + x) / 2 = 0.5) → (2 / y = 2 / y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_y_l2137_213769


namespace NUMINAMATH_CALUDE_segment_length_product_l2137_213789

theorem segment_length_product (a : ℝ) : 
  (∃ b : ℝ, b ≠ a ∧ 
   ((3 * a - 5)^2 + (2 * a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2) ∧
   ((3 * b - 5)^2 + (2 * b - 5 - (-2))^2 = (3 * Real.sqrt 13)^2)) →
  (a * b = -1080 / 169) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l2137_213789


namespace NUMINAMATH_CALUDE_roots_transformation_l2137_213743

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 4*x^2 + 5

-- Define the roots of the original polynomial
def roots_original : Set ℝ := {r | original_poly r = 0}

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 - 12*x^2 + 135

-- Define the roots of the new polynomial
def roots_new : Set ℝ := {r | new_poly r = 0}

-- State the theorem
theorem roots_transformation :
  ∃ (r₁ r₂ r₃ : ℝ), roots_original = {r₁, r₂, r₃} →
    roots_new = {3*r₁, 3*r₂, 3*r₃} :=
sorry

end NUMINAMATH_CALUDE_roots_transformation_l2137_213743


namespace NUMINAMATH_CALUDE_park_area_l2137_213773

/-- The area of a rectangular park given its length-to-breadth ratio and cycling time around its perimeter -/
theorem park_area (L B : ℝ) (h1 : L / B = 1 / 3) 
  (h2 : 2 * (L + B) = (12 * 1000 / 60) * 4) : L * B = 30000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l2137_213773


namespace NUMINAMATH_CALUDE_max_value_abc_l2137_213774

theorem max_value_abc (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_3 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → a + b^2 + c^4 ≤ x + y^2 + z^4 ∧ a + b^2 + c^4 ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2137_213774


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_product_l2137_213737

theorem sum_of_squares_geq_product (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ x₁ * (x₂ + x₃ + x₄ + x₅) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_product_l2137_213737


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2137_213750

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l2137_213750


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2137_213725

-- Problem 1
theorem problem_1 : (π - 3.14)^0 + Real.sqrt 16 + |1 - Real.sqrt 2| = 4 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : ∃ (x y : ℝ), x - y = 2 ∧ 2*x + 3*y = 9 ∧ x = 3 ∧ y = 1 := by sorry

-- Problem 3
theorem problem_3 : ∃ (x y : ℝ), 5*(x-1) + 2*y = 4*(1-y) + 3 ∧ x/3 + y/2 = 1 ∧ x = 0 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2137_213725


namespace NUMINAMATH_CALUDE_min_value_fraction_l2137_213756

theorem min_value_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2137_213756


namespace NUMINAMATH_CALUDE_school_paper_usage_theorem_l2137_213770

/-- The number of sheets of paper used by a school in a week -/
def school_paper_usage (sheets_per_class_per_day : ℕ) (school_days_per_week : ℕ) (num_classes : ℕ) : ℕ :=
  sheets_per_class_per_day * school_days_per_week * num_classes

/-- Theorem stating that under given conditions, the school uses 9000 sheets of paper per week -/
theorem school_paper_usage_theorem :
  school_paper_usage 200 5 9 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_school_paper_usage_theorem_l2137_213770


namespace NUMINAMATH_CALUDE_insufficient_info_for_unique_height_l2137_213794

/-- Represents the relationship between height and shadow length -/
noncomputable def height_shadow_relation (a b : ℝ) (s : ℝ) : ℝ :=
  a * Real.sqrt s + b * s

theorem insufficient_info_for_unique_height :
  ∀ (a₁ b₁ a₂ b₂ : ℝ),
  (height_shadow_relation a₁ b₁ 40.25 = 17.5) →
  (height_shadow_relation a₂ b₂ 40.25 = 17.5) →
  (a₁ ≠ a₂ ∨ b₁ ≠ b₂) →
  ∃ (h₁ h₂ : ℝ), 
    h₁ ≠ h₂ ∧ 
    height_shadow_relation a₁ b₁ 28.75 = h₁ ∧
    height_shadow_relation a₂ b₂ 28.75 = h₂ :=
by sorry

end NUMINAMATH_CALUDE_insufficient_info_for_unique_height_l2137_213794


namespace NUMINAMATH_CALUDE_largest_integer_l2137_213700

theorem largest_integer (a b c d : ℤ) 
  (sum_abc : a + b + c = 160)
  (sum_abd : a + b + d = 185)
  (sum_acd : a + c + d = 205)
  (sum_bcd : b + c + d = 230) :
  max a (max b (max c d)) = 100 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_l2137_213700


namespace NUMINAMATH_CALUDE_min_posts_for_given_area_l2137_213747

/-- Calculates the minimum number of fence posts required for a rectangular grazing area -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let perimeter := 2 * length + width
  let num_intervals := perimeter / post_spacing
  num_intervals + 1

theorem min_posts_for_given_area : 
  min_fence_posts 80 40 10 = 17 :=
by
  sorry

#eval min_fence_posts 80 40 10

end NUMINAMATH_CALUDE_min_posts_for_given_area_l2137_213747


namespace NUMINAMATH_CALUDE_probability_of_circle_l2137_213703

theorem probability_of_circle (total_figures : ℕ) (circles : ℕ) 
  (h1 : total_figures = 10) 
  (h2 : circles = 4) : 
  (circles : ℚ) / total_figures = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_circle_l2137_213703


namespace NUMINAMATH_CALUDE_inequality_solutions_m_range_l2137_213701

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem for Question I
theorem inequality_solutions (a : ℝ) :
  (a = 1 → {x : ℝ | f x > 0} = {x : ℝ | x < 2 ∨ x > 2}) ∧
  (a > 1 → {x : ℝ | f x + a - 1 > 0} = Set.univ) ∧
  (a < 1 → {x : ℝ | f x + a - 1 > 0} = {x : ℝ | x < a + 1 ∨ x > 3 - a}) :=
sorry

-- Theorem for Question II
theorem m_range (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solutions_m_range_l2137_213701


namespace NUMINAMATH_CALUDE_random_events_identification_l2137_213731

-- Define the type for events
inductive Event
| CoinToss : Event
| ChargeAttraction : Event
| WaterFreeze : Event
| DieRoll : Event

-- Define a predicate for random events
def isRandomEvent : Event → Prop
| Event.CoinToss => true
| Event.ChargeAttraction => false
| Event.WaterFreeze => false
| Event.DieRoll => true

-- Theorem statement
theorem random_events_identification :
  (isRandomEvent Event.CoinToss ∧ isRandomEvent Event.DieRoll) ∧
  (¬isRandomEvent Event.ChargeAttraction ∧ ¬isRandomEvent Event.WaterFreeze) :=
by sorry

end NUMINAMATH_CALUDE_random_events_identification_l2137_213731


namespace NUMINAMATH_CALUDE_alberts_earnings_increase_l2137_213712

theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  (1.26 * E = 693) → 
  ((1 + P) * E = 660) →
  P = 0.2 := by
sorry

end NUMINAMATH_CALUDE_alberts_earnings_increase_l2137_213712


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2137_213795

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2137_213795


namespace NUMINAMATH_CALUDE_valid_arrangements_l2137_213749

/-- The number of ways to arrange 2 black, 3 white, and 4 red balls in a row such that no black ball is next to a white ball. -/
def arrangeBalls : ℕ := 200

/-- The number of black balls -/
def blackBalls : ℕ := 2

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of red balls -/
def redBalls : ℕ := 4

/-- Theorem stating that the number of valid arrangements is equal to arrangeBalls -/
theorem valid_arrangements :
  (∃ (f : ℕ → ℕ → ℕ → ℕ), f blackBalls whiteBalls redBalls = arrangeBalls) :=
sorry

end NUMINAMATH_CALUDE_valid_arrangements_l2137_213749


namespace NUMINAMATH_CALUDE_laborer_wage_calculation_l2137_213796

/-- The daily wage of a general laborer -/
def laborer_wage : ℕ :=
  -- Define the wage here
  sorry

/-- The number of people hired -/
def total_hired : ℕ := 31

/-- The total payroll in dollars -/
def total_payroll : ℕ := 3952

/-- The daily wage of a heavy operator -/
def operator_wage : ℕ := 129

/-- The number of laborers employed -/
def laborers_employed : ℕ := 1

theorem laborer_wage_calculation : 
  laborer_wage = 82 ∧
  total_hired * operator_wage - (total_hired - laborers_employed) * operator_wage + laborer_wage = total_payroll :=
by sorry

end NUMINAMATH_CALUDE_laborer_wage_calculation_l2137_213796


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l2137_213787

theorem pasta_preference_ratio (total_students : ℕ) 
  (spaghetti_preference : ℕ) (fettuccine_preference : ℕ) :
  total_students = 800 →
  spaghetti_preference = 300 →
  fettuccine_preference = 80 →
  (spaghetti_preference : ℚ) / fettuccine_preference = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l2137_213787


namespace NUMINAMATH_CALUDE_susan_gave_out_half_apples_l2137_213786

theorem susan_gave_out_half_apples (frank_apples : ℕ) (susan_apples : ℕ) (total_left : ℕ) :
  frank_apples = 36 →
  susan_apples = 3 * frank_apples →
  total_left = 78 →
  total_left = susan_apples + frank_apples - frank_apples / 3 - susan_apples * (1 - x) →
  x = (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_susan_gave_out_half_apples_l2137_213786


namespace NUMINAMATH_CALUDE_seeds_per_row_l2137_213761

/-- Given a garden with potatoes planted in rows, this theorem proves
    the number of seeds in each row when the total number of potatoes
    and the number of rows are known. -/
theorem seeds_per_row (total_potatoes : ℕ) (num_rows : ℕ) 
    (h1 : total_potatoes = 54) 
    (h2 : num_rows = 6) 
    (h3 : total_potatoes % num_rows = 0) : 
  total_potatoes / num_rows = 9 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_row_l2137_213761


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2137_213754

theorem isosceles_triangle_vertex_angle (α : ℝ) :
  α > 0 ∧ α < 180 →  -- Angle is positive and less than 180°
  50 > 0 ∧ 50 < 180 →  -- 50° is a valid angle
  α + 50 + 50 = 180 →  -- Sum of angles in a triangle is 180°
  α = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l2137_213754


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l2137_213788

theorem reciprocal_of_2023 :
  let reciprocal (x : ℝ) := 1 / x
  reciprocal 2023 = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l2137_213788


namespace NUMINAMATH_CALUDE_caravan_hens_l2137_213759

def caravan (num_hens : ℕ) : Prop :=
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := 2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers
  total_feet = total_heads + 224

theorem caravan_hens : ∃ (num_hens : ℕ), caravan num_hens ∧ num_hens = 50 := by
  sorry

end NUMINAMATH_CALUDE_caravan_hens_l2137_213759


namespace NUMINAMATH_CALUDE_ocean_area_ratio_l2137_213710

theorem ocean_area_ratio (total_area land_area ocean_area : ℝ)
  (land_ratio : land_area / total_area = 29 / 100)
  (ocean_ratio : ocean_area / total_area = 71 / 100)
  (northern_land : ℝ) (southern_land : ℝ)
  (northern_land_ratio : northern_land / land_area = 3 / 4)
  (southern_land_ratio : southern_land / land_area = 1 / 4)
  (northern_ocean southern_ocean : ℝ)
  (northern_hemisphere : northern_land + northern_ocean = total_area / 2)
  (southern_hemisphere : southern_land + southern_ocean = total_area / 2) :
  southern_ocean / northern_ocean = 171 / 113 :=
sorry

end NUMINAMATH_CALUDE_ocean_area_ratio_l2137_213710


namespace NUMINAMATH_CALUDE_abs_difference_implies_abs_inequality_l2137_213766

theorem abs_difference_implies_abs_inequality (a_n l : ℝ) 
  (h : |a_n - l| > 1) : |a_n| > 1 - |l| := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_implies_abs_inequality_l2137_213766


namespace NUMINAMATH_CALUDE_birds_in_marsh_l2137_213790

theorem birds_in_marsh (geese ducks : ℕ) (h1 : geese = 58) (h2 : ducks = 37) :
  geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_l2137_213790


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2137_213719

/-- The surface area of a sphere circumscribing a cube with side length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube :
  let cube_side_length : ℝ := 1
  let sphere_radius : ℝ := Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2137_213719


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l2137_213735

-- Define the function
def f (x : ℝ) : ℝ := (x - 3) * |x|

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- Theorem statement
theorem f_decreasing_interval :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  is_decreasing_on f a b ∧
  ∀ (c d : ℝ), c < a ∨ b < d → ¬(is_decreasing_on f c d) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l2137_213735


namespace NUMINAMATH_CALUDE_inequality_implication_l2137_213760

theorem inequality_implication (x y : ℝ) : x < y → -x/2 > -y/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2137_213760


namespace NUMINAMATH_CALUDE_smallest_n_doughnuts_l2137_213730

theorem smallest_n_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (15 * m - 1) % 5 = 0 → m ≥ n) ∧
  (15 * n - 1) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_doughnuts_l2137_213730


namespace NUMINAMATH_CALUDE_earthquake_relief_donation_scientific_notation_l2137_213755

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem earthquake_relief_donation_scientific_notation :
  toScientificNotation 3990000000 = ScientificNotation.mk 3.99 9 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_earthquake_relief_donation_scientific_notation_l2137_213755


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2137_213707

theorem algebraic_simplification (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2137_213707


namespace NUMINAMATH_CALUDE_students_walking_home_l2137_213724

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction : ℚ)
  (h1 : bus_fraction = 1/2)
  (h2 : automobile_fraction = 1/4)
  (h3 : bicycle_fraction = 1/10) :
  1 - (bus_fraction + automobile_fraction + bicycle_fraction) = 3/20 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2137_213724


namespace NUMINAMATH_CALUDE_puzzle_solution_l2137_213744

-- Define the types of beings
inductive Being
| Human
| Monkey

-- Define the types of speakers
inductive Speaker
| Knight
| Liar

-- Define A and B as individuals
structure Individual where
  being : Being
  speaker : Speaker

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.being = Being.Monkey ∨ b.being = Being.Monkey

def statement_B (a b : Individual) : Prop :=
  a.speaker = Speaker.Liar ∨ b.speaker = Speaker.Liar

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.speaker = Speaker.Liar) ∧
    (statement_B a b ↔ b.speaker = Speaker.Knight) ∧
    a.being = Being.Human ∧
    b.being = Being.Human ∧
    a.speaker = Speaker.Liar ∧
    b.speaker = Speaker.Knight :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2137_213744


namespace NUMINAMATH_CALUDE_parabola_vertex_l2137_213740

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 2)^2 - 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -5)

/-- Theorem: The vertex of the parabola y = -2(x-2)^2 - 5 is at the point (2, -5) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2137_213740


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l2137_213798

theorem smallest_of_three_consecutive_odds (x y z : ℤ) : 
  (∃ k : ℤ, x = 2*k + 1) →  -- x is odd
  y = x + 2 →               -- y is the next consecutive odd number
  z = y + 2 →               -- z is the next consecutive odd number after y
  x + y + z = 69 →          -- their sum is 69
  x = 21                    -- x (the smallest) is 21
:= by sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l2137_213798


namespace NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l2137_213772

theorem geometric_progression_sum_ratio (m : ℕ) : 
  let r : ℝ := 3
  let S (n : ℕ) := (1 - r^n) / (1 - r)
  (S 6) / (S m) = 28 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_ratio_l2137_213772


namespace NUMINAMATH_CALUDE_double_time_double_discount_l2137_213745

/-- Represents the true discount calculation for a bill -/
def true_discount (face_value : ℝ) (discount : ℝ) (time : ℝ) : Prop :=
  ∃ (rate : ℝ),
    discount = (face_value - discount) * rate * time ∧
    rate > 0 ∧
    time > 0

/-- 
If the true discount on a bill of 110 is 10 for a certain time,
then the true discount on the same bill for double the time is 20.
-/
theorem double_time_double_discount :
  ∀ (time : ℝ),
    true_discount 110 10 time →
    true_discount 110 20 (2 * time) :=
by
  sorry

end NUMINAMATH_CALUDE_double_time_double_discount_l2137_213745


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2137_213718

theorem fraction_sum_equality (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2137_213718


namespace NUMINAMATH_CALUDE_not_multiple_of_three_l2137_213739

theorem not_multiple_of_three (n : ℕ) (h : ∃ m : ℕ, n * (n + 3) = m ^ 2) : ¬ (3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_multiple_of_three_l2137_213739


namespace NUMINAMATH_CALUDE_jaco_gift_budget_l2137_213799

/-- Calculates the budget for each parent's gift given a total budget, number of friends, 
    cost per friend's gift, and number of parents. -/
def parent_gift_budget (total_budget : ℚ) (num_friends : ℕ) (friend_gift_cost : ℚ) (num_parents : ℕ) : ℚ :=
  (total_budget - (num_friends : ℚ) * friend_gift_cost) / (num_parents : ℚ)

/-- Proves that given a total budget of $100, 8 friends' gifts costing $9 each, 
    and equal-cost gifts for two parents, the budget for each parent's gift is $14. -/
theorem jaco_gift_budget : parent_gift_budget 100 8 9 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jaco_gift_budget_l2137_213799


namespace NUMINAMATH_CALUDE_move_negative_one_right_three_l2137_213777

-- Define the movement on a number line
def move_right (start : ℤ) (distance : ℕ) : ℤ := start + distance

-- Theorem statement
theorem move_negative_one_right_three : move_right (-1) 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_move_negative_one_right_three_l2137_213777


namespace NUMINAMATH_CALUDE_mushroom_price_per_unit_l2137_213797

theorem mushroom_price_per_unit (total_mushrooms day2_mushrooms day1_revenue : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  day1_revenue = 58 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * 2 = day1_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_mushroom_price_per_unit_l2137_213797


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l2137_213705

theorem fixed_point_power_function (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2)^a + 1
  f 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l2137_213705


namespace NUMINAMATH_CALUDE_sphere_sum_l2137_213714

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_sum_l2137_213714


namespace NUMINAMATH_CALUDE_bezout_unique_solution_l2137_213733

theorem bezout_unique_solution (a b : ℕ) (ha : a > 1) (hb : b > 1) (hgcd : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), a * r - b * s = 1 ∧ 0 < r ∧ r < b ∧ 0 < s ∧ s < a :=
sorry

end NUMINAMATH_CALUDE_bezout_unique_solution_l2137_213733


namespace NUMINAMATH_CALUDE_unused_types_count_l2137_213763

/-- The number of natural resources --/
def num_resources : ℕ := 6

/-- The number of types of nature use developed --/
def types_developed : ℕ := 23

/-- The total number of possible combinations of resource usage --/
def total_combinations : ℕ := 2^num_resources

/-- The number of valid combinations (excluding the all-zero combination) --/
def valid_combinations : ℕ := total_combinations - 1

/-- The number of unused types of nature use --/
def unused_types : ℕ := valid_combinations - types_developed

theorem unused_types_count : unused_types = 40 := by
  sorry

end NUMINAMATH_CALUDE_unused_types_count_l2137_213763


namespace NUMINAMATH_CALUDE_morning_campers_l2137_213728

theorem morning_campers (total : ℕ) (afternoon : ℕ) (morning : ℕ) : 
  total = 60 → afternoon = 7 → morning = total - afternoon → morning = 53 := by
sorry

end NUMINAMATH_CALUDE_morning_campers_l2137_213728


namespace NUMINAMATH_CALUDE_two_ducks_in_garden_l2137_213775

/-- The number of ducks in a garden with dogs and ducks -/
def number_of_ducks (num_dogs : ℕ) (total_feet : ℕ) : ℕ :=
  (total_feet - 4 * num_dogs) / 2

/-- Theorem: There are 2 ducks in the garden -/
theorem two_ducks_in_garden : number_of_ducks 6 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_ducks_in_garden_l2137_213775


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l2137_213757

/-- Represents the number of people to be selected from each group in a stratified sampling --/
structure StratifiedSample where
  regular : ℕ
  middle : ℕ
  senior : ℕ

/-- Calculates the stratified sample given total employees and managers --/
def calculateStratifiedSample (total : ℕ) (middle : ℕ) (senior : ℕ) (toSelect : ℕ) : StratifiedSample :=
  sorry

/-- Theorem stating that the calculated stratified sample is correct --/
theorem stratified_sample_correct :
  let total := 160
  let middle := 30
  let senior := 10
  let toSelect := 20
  let result := calculateStratifiedSample total middle senior toSelect
  result.regular = 16 ∧ result.middle = 3 ∧ result.senior = 1 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_correct_l2137_213757


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l2137_213738

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l2137_213738


namespace NUMINAMATH_CALUDE_slope_positive_for_a_in_open_unit_interval_l2137_213736

theorem slope_positive_for_a_in_open_unit_interval :
  ∀ a : ℝ, 0 < a ∧ a < 1 →
  let k := -(2^a - 1) / Real.log a
  k > 0 := by
sorry

end NUMINAMATH_CALUDE_slope_positive_for_a_in_open_unit_interval_l2137_213736


namespace NUMINAMATH_CALUDE_coin_jar_problem_l2137_213762

/-- Represents the contents and value of a jar of coins. -/
structure CoinJar where
  pennies : ℕ
  nickels : ℕ
  quarters : ℕ
  total_coins : ℕ
  total_value : ℚ

/-- Theorem stating the conditions and result for the coin jar problem. -/
theorem coin_jar_problem (jar : CoinJar) : 
  jar.nickels = 3 * jar.pennies →
  jar.quarters = 4 * jar.nickels →
  jar.total_coins = jar.pennies + jar.nickels + jar.quarters →
  jar.total_coins = 240 →
  jar.total_value = jar.pennies * (1 : ℚ) / 100 + 
                    jar.nickels * (5 : ℚ) / 100 + 
                    jar.quarters * (25 : ℚ) / 100 →
  jar.total_value = (4740 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_jar_problem_l2137_213762


namespace NUMINAMATH_CALUDE_inequality_proof_l2137_213726

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  a * (1 + b - c)^(1/3 : ℝ) + b * (1 + c - a)^(1/3 : ℝ) + c * (1 + a - b)^(1/3 : ℝ) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2137_213726


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2137_213752

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^3 * x^(1/2))^(1/4) = x^(7/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2137_213752


namespace NUMINAMATH_CALUDE_triangle_angle_value_l2137_213768

theorem triangle_angle_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c →
  B = π / 3 ∨ B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l2137_213768


namespace NUMINAMATH_CALUDE_circle_radius_determines_m_l2137_213784

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 - 2*h*x - 2*k*y + (h^2 + k^2 - r^2 + m) = 0

theorem circle_radius_determines_m :
  ∀ m : ℝ, (∃ h k : ℝ, is_circle_equation h k 2 m) → m = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_determines_m_l2137_213784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2137_213711

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 6 + a 11 = 3 →
  a 3 + a 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2137_213711


namespace NUMINAMATH_CALUDE_paint_house_theorem_l2137_213751

/-- Represents the time (in hours) it takes to paint a house given the number of people working -/
def paintTime (people : ℕ) : ℚ :=
  24 / people

theorem paint_house_theorem :
  paintTime 4 = 6 →
  paintTime 3 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_house_theorem_l2137_213751


namespace NUMINAMATH_CALUDE_harry_hours_formula_l2137_213765

/-- Represents the payment structure and hours worked for Harry and James -/
structure PaymentSystem where
  x : ℝ  -- Base hourly rate
  S : ℝ  -- Number of hours James is paid at regular rate
  H : ℝ  -- Number of hours Harry worked

/-- Calculates Harry's pay for the week -/
def harry_pay (p : PaymentSystem) : ℝ :=
  18 * p.x + 1.5 * p.x * (p.H - 18)

/-- Calculates James' pay for the week -/
def james_pay (p : PaymentSystem) : ℝ :=
  p.S * p.x + 2 * p.x * (41 - p.S)

/-- Theorem stating the relationship between Harry's hours worked and James' regular hours -/
theorem harry_hours_formula (p : PaymentSystem) :
  harry_pay p = james_pay p →
  p.H = (91 - 3 * p.S) / 1.5 := by
  sorry

end NUMINAMATH_CALUDE_harry_hours_formula_l2137_213765


namespace NUMINAMATH_CALUDE_poster_wall_width_l2137_213702

/-- The minimum width of wall needed to attach posters -/
def minimum_wall_width (poster_width : ℕ) (overlap : ℕ) (num_posters : ℕ) : ℕ :=
  poster_width + (num_posters - 1) * (poster_width - overlap)

/-- Theorem stating the minimum wall width for given poster specifications -/
theorem poster_wall_width :
  minimum_wall_width 30 2 15 = 422 := by
  sorry

end NUMINAMATH_CALUDE_poster_wall_width_l2137_213702


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2137_213748

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a4a6 : a 4 * a 6 = 64) :
  ∃ q : ℝ, is_geometric_sequence a ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2137_213748


namespace NUMINAMATH_CALUDE_second_track_has_30_checkpoints_l2137_213721

/-- The number of checkpoints on the first track -/
def first_track_checkpoints : ℕ := 6

/-- The total number of ways to form triangles -/
def total_triangles : ℕ := 420

/-- The number of checkpoints on the second track -/
def second_track_checkpoints : ℕ := 30

/-- Theorem stating that the number of checkpoints on the second track is 30 -/
theorem second_track_has_30_checkpoints :
  (first_track_checkpoints * (second_track_checkpoints.choose 2) = total_triangles) →
  second_track_checkpoints = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_track_has_30_checkpoints_l2137_213721


namespace NUMINAMATH_CALUDE_difference_of_integers_l2137_213764

/-- Given positive integers a and b satisfying 2a - 9b + 18ab = 2018, prove that b - a = 223 -/
theorem difference_of_integers (a b : ℕ+) (h : 2 * a - 9 * b + 18 * a * b = 2018) : 
  b - a = 223 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_integers_l2137_213764


namespace NUMINAMATH_CALUDE_apple_baskets_proof_l2137_213727

/-- Given a total number of apples and apples per basket, calculate the number of full baskets -/
def fullBaskets (totalApples applesPerBasket : ℕ) : ℕ :=
  totalApples / applesPerBasket

theorem apple_baskets_proof :
  fullBaskets 495 25 = 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_baskets_proof_l2137_213727


namespace NUMINAMATH_CALUDE_expression_evaluation_l2137_213779

theorem expression_evaluation (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2137_213779


namespace NUMINAMATH_CALUDE_jerrys_books_l2137_213723

/-- Given Jerry's initial and additional books, prove the total number of books. -/
theorem jerrys_books (initial_books additional_books : ℕ) :
  initial_books = 9 → additional_books = 10 → initial_books + additional_books = 19 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_books_l2137_213723


namespace NUMINAMATH_CALUDE_employee_bonuses_l2137_213791

theorem employee_bonuses :
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = 2970 ∧
    y = (1/3) * x + 180 ∧
    z = (1/3) * y + 130 ∧
    x = 1800 ∧ y = 780 ∧ z = 390 := by
  sorry

end NUMINAMATH_CALUDE_employee_bonuses_l2137_213791


namespace NUMINAMATH_CALUDE_wood_square_weight_l2137_213715

/-- Represents a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Calculates the area of a square -/
def square_area (s : ℝ) : ℝ := s * s

/-- Theorem: Given two square pieces of wood with uniform density, 
    where the first piece has a side length of 3 inches and weighs 15 ounces, 
    and the second piece has a side length of 6 inches, 
    the weight of the second piece is 60 ounces. -/
theorem wood_square_weight 
  (first : WoodSquare) 
  (second : WoodSquare) 
  (h1 : first.side_length = 3) 
  (h2 : first.weight = 15) 
  (h3 : second.side_length = 6) : 
  second.weight = 60 := by
  sorry


end NUMINAMATH_CALUDE_wood_square_weight_l2137_213715


namespace NUMINAMATH_CALUDE_not_p_and_q_is_true_l2137_213741

theorem not_p_and_q_is_true (h1 : ¬(p ∧ q)) (h2 : ¬¬q) : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_is_true_l2137_213741


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2137_213713

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2137_213713


namespace NUMINAMATH_CALUDE_sin_triple_angle_l2137_213708

theorem sin_triple_angle (θ : ℝ) :
  Real.sin (3 * θ) = 4 * Real.sin θ * Real.sin (π / 3 + θ) * Real.sin (2 * π / 3 + θ) := by
  sorry

end NUMINAMATH_CALUDE_sin_triple_angle_l2137_213708


namespace NUMINAMATH_CALUDE_brian_pencils_given_to_friend_l2137_213753

/-- 
Given that Brian initially had 39 pencils, bought 22 more, and ended up with 43 pencils,
this theorem proves that Brian gave 18 pencils to his friend.
-/
theorem brian_pencils_given_to_friend : 
  ∀ (initial_pencils bought_pencils final_pencils pencils_given : ℕ),
    initial_pencils = 39 →
    bought_pencils = 22 →
    final_pencils = 43 →
    final_pencils = initial_pencils - pencils_given + bought_pencils →
    pencils_given = 18 := by
  sorry

end NUMINAMATH_CALUDE_brian_pencils_given_to_friend_l2137_213753


namespace NUMINAMATH_CALUDE_pradeep_failed_by_25_marks_l2137_213722

/-- Calculates the number of marks by which a student failed, given the total marks,
    passing percentage, and the student's marks. -/
def marksFailed (totalMarks passingPercentage studentMarks : ℕ) : ℕ :=
  let passingMarks := totalMarks * passingPercentage / 100
  if studentMarks ≥ passingMarks then 0
  else passingMarks - studentMarks

/-- Theorem stating that Pradeep failed by 25 marks -/
theorem pradeep_failed_by_25_marks :
  marksFailed 840 25 185 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pradeep_failed_by_25_marks_l2137_213722


namespace NUMINAMATH_CALUDE_pencils_per_row_l2137_213771

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 32)
  (h2 : num_rows = 4)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l2137_213771


namespace NUMINAMATH_CALUDE_distinct_groups_eq_seven_l2137_213782

/-- The number of distinct groups of 3 marbles Tom can choose -/
def distinct_groups : ℕ :=
  let red_marbles : ℕ := 1
  let green_marbles : ℕ := 1
  let blue_marbles : ℕ := 1
  let yellow_marbles : ℕ := 4
  let non_yellow_marbles : ℕ := red_marbles + green_marbles + blue_marbles
  let all_yellow_groups : ℕ := 1
  let two_yellow_groups : ℕ := non_yellow_marbles
  let one_yellow_groups : ℕ := Nat.choose non_yellow_marbles 2
  all_yellow_groups + two_yellow_groups + one_yellow_groups

theorem distinct_groups_eq_seven : distinct_groups = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_groups_eq_seven_l2137_213782


namespace NUMINAMATH_CALUDE_debate_team_formations_l2137_213704

def num_boys : ℕ := 3
def num_girls : ℕ := 3
def num_debaters : ℕ := 4
def boy_a_exists : Prop := true

theorem debate_team_formations :
  (num_boys + num_girls - 1) * (num_boys + num_girls - 1) * (num_boys + num_girls - 2) * (num_boys + num_girls - 3) = 300 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_formations_l2137_213704


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2137_213742

theorem circle_center_and_radius :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 5)^2 = 3 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -5) ∧ radius = Real.sqrt 3 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2137_213742


namespace NUMINAMATH_CALUDE_eighth_root_of_5487587353601_l2137_213706

theorem eighth_root_of_5487587353601 : ∃ n : ℕ, n ^ 8 = 5487587353601 ∧ n = 101 := by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_5487587353601_l2137_213706


namespace NUMINAMATH_CALUDE_sqrt_sum_value_l2137_213709

theorem sqrt_sum_value (a b : ℝ) (h : (Real.sqrt a + Real.sqrt b) * (Real.sqrt a + Real.sqrt b - 2) = 3) :
  Real.sqrt a + Real.sqrt b = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_value_l2137_213709


namespace NUMINAMATH_CALUDE_total_blocks_adolfos_blocks_l2137_213716

/-- Given an initial number of blocks and a number of blocks added, 
    the total number of blocks is equal to the sum of the initial blocks and added blocks. -/
theorem total_blocks (initial_blocks added_blocks : ℕ) :
  initial_blocks + added_blocks = initial_blocks + added_blocks := by
  sorry

/-- Adolfo's block problem -/
theorem adolfos_blocks : 
  let initial_blocks : ℕ := 35
  let added_blocks : ℕ := 30
  initial_blocks + added_blocks = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_adolfos_blocks_l2137_213716


namespace NUMINAMATH_CALUDE_females_in_coach_class_l2137_213785

theorem females_in_coach_class 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (first_class_male_ratio : ℚ) 
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 45/100)
  (h3 : first_class_percentage = 10/100)
  (h4 : first_class_male_ratio = 1/3) :
  (total_passengers : ℚ) * female_percentage - 
  (total_passengers : ℚ) * first_class_percentage * (1 - first_class_male_ratio) = 46 := by
sorry

end NUMINAMATH_CALUDE_females_in_coach_class_l2137_213785


namespace NUMINAMATH_CALUDE_distance_between_trees_l2137_213780

/-- Given a yard of length 500 metres with 105 trees planted at equal distances,
    including one at each end, prove that the distance between two consecutive
    trees is 500/104 metres. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 500)
    (h2 : num_trees = 105) :
  let num_segments := num_trees - 1
  yard_length / num_segments = 500 / 104 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2137_213780


namespace NUMINAMATH_CALUDE_student_assignment_count_l2137_213717

/-- The number of ways to assign students to internship positions -/
def assignment_count (n_students : ℕ) (n_positions : ℕ) : ℕ :=
  (n_students.choose 2) * (n_positions.factorial)

/-- Theorem: There are 36 ways to assign 4 students to 3 internship positions -/
theorem student_assignment_count :
  assignment_count 4 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_assignment_count_l2137_213717


namespace NUMINAMATH_CALUDE_work_payment_theorem_l2137_213792

/-- Represents the time (in days) it takes for a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_pos : days > 0

/-- Represents a worker with their work time and share of payment -/
structure Worker where
  work_time : WorkTime
  share : ℚ
  share_nonneg : share ≥ 0

/-- Calculates the total payment for a job given two workers' information -/
def total_payment (worker1 worker2 : Worker) : ℚ :=
  let total_work_rate := 1 / worker1.work_time.days + 1 / worker2.work_time.days
  let total_parts := total_work_rate * worker1.work_time.days * worker2.work_time.days
  let worker1_parts := worker2.work_time.days
  worker1.share * total_parts / worker1_parts

/-- The main theorem stating the total payment for the work -/
theorem work_payment_theorem (rahul rajesh : Worker) 
    (h1 : rahul.work_time.days = 3)
    (h2 : rajesh.work_time.days = 2)
    (h3 : rahul.share = 900) :
    total_payment rahul rajesh = 2250 := by
  sorry


end NUMINAMATH_CALUDE_work_payment_theorem_l2137_213792


namespace NUMINAMATH_CALUDE_mushroom_picking_profit_l2137_213793

/-- Calculates the money made on the first day of a three-day mushroom picking trip -/
theorem mushroom_picking_profit (total_mushrooms day2_mushrooms price_per_mushroom : ℕ) : 
  total_mushrooms = 65 →
  day2_mushrooms = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - day2_mushrooms - 2 * day2_mushrooms) * price_per_mushroom = 58 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_picking_profit_l2137_213793


namespace NUMINAMATH_CALUDE_news_program_selection_methods_l2137_213720

theorem news_program_selection_methods (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 4 → m = 2 →
  (n.choose k) * (k.choose m) * (m.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_news_program_selection_methods_l2137_213720


namespace NUMINAMATH_CALUDE_total_supermarkets_l2137_213783

def FGH_chain (us canada : ℕ) : Prop :=
  (us = 49) ∧ (us = canada + 14)

theorem total_supermarkets (us canada : ℕ) (h : FGH_chain us canada) : 
  us + canada = 84 :=
sorry

end NUMINAMATH_CALUDE_total_supermarkets_l2137_213783


namespace NUMINAMATH_CALUDE_chair_probability_l2137_213781

/- Define the number of chairs -/
def total_chairs : ℕ := 10

/- Define the number of broken chairs -/
def broken_chairs : ℕ := 2

/- Define the number of usable chairs -/
def usable_chairs : ℕ := total_chairs - broken_chairs

/- Define the number of adjacent pairs in usable chairs -/
def adjacent_pairs : ℕ := usable_chairs - 1 - 1  -- Subtract 1 for the gap between 4 and 7

/- Define the probability of not sitting next to each other -/
def prob_not_adjacent : ℚ := 11 / 14

theorem chair_probability : 
  prob_not_adjacent = 1 - (adjacent_pairs : ℚ) / (usable_chairs.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_chair_probability_l2137_213781


namespace NUMINAMATH_CALUDE_rectangle_sides_theorem_l2137_213729

/-- A pair of positive integers representing the sides of a rectangle --/
structure RectangleSides where
  x : ℕ+
  y : ℕ+

/-- The set of rectangle sides that satisfy the perimeter-area equality condition --/
def validRectangleSides : Set RectangleSides :=
  { sides | (2 * sides.x.val + 2 * sides.y.val : ℕ) = sides.x.val * sides.y.val }

/-- The theorem stating that only three specific pairs of sides satisfy the conditions --/
theorem rectangle_sides_theorem :
  validRectangleSides = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_theorem_l2137_213729


namespace NUMINAMATH_CALUDE_f_42_17_l2137_213758

def is_valid_f (f : ℚ → Int) : Prop :=
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → f x * f y = -1) ∧
  (∀ x : ℚ, f x = 1 ∨ f x = -1) ∧
  f 0 = 1

theorem f_42_17 (f : ℚ → Int) (h : is_valid_f f) : f (42/17) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_42_17_l2137_213758


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt_181_over_12_l2137_213732

theorem sqrt_sum_fractions_equals_sqrt_181_over_12 :
  Real.sqrt (9/16 + 25/36) = Real.sqrt 181 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt_181_over_12_l2137_213732


namespace NUMINAMATH_CALUDE_floor_equation_equivalence_l2137_213776

def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, (k + 1/5 ≤ x ∧ x < k + 1/3) ∨
           (k + 2/5 ≤ x ∧ x < k + 3/5) ∨
           (k + 2/3 ≤ x ∧ x < k + 4/5)

theorem floor_equation_equivalence (x : ℝ) :
  ⌊(5 : ℝ) * x⌋ = ⌊(3 : ℝ) * x⌋ + 2 * ⌊x⌋ + 1 ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_equivalence_l2137_213776


namespace NUMINAMATH_CALUDE_marble_problem_l2137_213767

theorem marble_problem (A V R S x : ℝ) 
  (h1 : A + x = V - x)
  (h2 : V + 2*x = A - 2*x + 30)
  (h3 : (A + R/2) + (V + R/2) = 120)
  (h4 : S - 0.25*S + 10 = 2*(R - R/2)) :
  x = 5 := by sorry

end NUMINAMATH_CALUDE_marble_problem_l2137_213767


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l2137_213734

-- Define a quadratic polynomial with real coefficients
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_zero 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, QuadraticPolynomial a b c (x^3 - x) ≥ QuadraticPolynomial a b c (x^2 - 1)) : 
  (- b) / a = 0 := by
  sorry

#check sum_of_roots_zero

end NUMINAMATH_CALUDE_sum_of_roots_zero_l2137_213734


namespace NUMINAMATH_CALUDE_problem_solution_l2137_213746

theorem problem_solution : 
  let P : ℚ := 4050 / 5
  let Q : ℚ := P / 4
  let Y : ℚ := P - Q
  Y = 607.5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2137_213746
