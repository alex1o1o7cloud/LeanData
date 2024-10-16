import Mathlib

namespace NUMINAMATH_CALUDE_jellybean_problem_l4070_407095

theorem jellybean_problem : ∃ (n : ℕ), 
  (n ≥ 150) ∧ 
  (n % 15 = 14) ∧ 
  (∀ m : ℕ, m ≥ 150 → m % 15 = 14 → m ≥ n) ∧
  (n = 164) := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l4070_407095


namespace NUMINAMATH_CALUDE_unique_solution_system_l4070_407083

theorem unique_solution_system (x y u v : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ u > 0 ∧ v > 0)
  (h1 : x + y = u)
  (h2 : v * x * y = u + v)
  (h3 : x * y * u * v = 16) :
  x = 2 ∧ y = 2 ∧ u = 2 ∧ v = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l4070_407083


namespace NUMINAMATH_CALUDE_work_completion_time_l4070_407067

theorem work_completion_time (x : ℝ) 
  (hx_pos : x > 0) 
  (hb_time : 20 > 0) 
  (htotal_time : 10 > 0) 
  (ha_left : 5 > 0) :
  (5 * (1/x + 1/20) + 5 * (1/20) = 1) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4070_407067


namespace NUMINAMATH_CALUDE_crayon_selection_l4070_407000

theorem crayon_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  (Nat.choose (n - 1) (k - 1)) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_l4070_407000


namespace NUMINAMATH_CALUDE_all_statements_imply_theorem_l4070_407041

theorem all_statements_imply_theorem (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) ∨ (¬p ∧ ¬q ∧ r) ∨ (p ∧ ¬q ∧ ¬r) ∨ (¬p ∧ q ∧ r)) → ((p → q) → r) := by
  sorry

#check all_statements_imply_theorem

end NUMINAMATH_CALUDE_all_statements_imply_theorem_l4070_407041


namespace NUMINAMATH_CALUDE_coat_final_price_coat_price_is_81_l4070_407002

/-- The final price of a coat after discounts and tax -/
theorem coat_final_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (additional_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + tax_rate)
  final_price

/-- Proof that the final price of the coat is $81 -/
theorem coat_price_is_81 : 
  coat_final_price 100 0.2 5 0.08 = 81 := by
  sorry

end NUMINAMATH_CALUDE_coat_final_price_coat_price_is_81_l4070_407002


namespace NUMINAMATH_CALUDE_car_travel_distance_l4070_407001

/-- Proves that Car X travels 98 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start_time : ℝ) : 
  speed_x = 35 →
  speed_y = 50 →
  head_start_time = 72 / 60 →
  ∃ (travel_time : ℝ), 
    travel_time > 0 ∧
    speed_x * (head_start_time + travel_time) = speed_y * travel_time ∧
    speed_x * travel_time = 98 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l4070_407001


namespace NUMINAMATH_CALUDE_tricycles_in_garage_l4070_407044

/-- The number of tricycles in Zoe's garage --/
def num_tricycles : ℕ := sorry

/-- The total number of wheels in the garage --/
def total_wheels : ℕ := 25

/-- The number of bicycles in the garage --/
def num_bicycles : ℕ := 3

/-- The number of unicycles in the garage --/
def num_unicycles : ℕ := 7

/-- The number of wheels on a bicycle --/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle --/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle --/
def wheels_per_unicycle : ℕ := 1

/-- Theorem stating that there are 4 tricycles in the garage --/
theorem tricycles_in_garage : num_tricycles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tricycles_in_garage_l4070_407044


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4070_407098

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 2*x) ↔ (∀ x : ℝ, x^2 + 1 ≥ 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4070_407098


namespace NUMINAMATH_CALUDE_max_x0_value_l4070_407097

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h_pos : ∀ i, x i > 0)
  (h_cycle : x 0 = x 1995)
  (h_relation : ∀ i : Fin 1995, x (i + 1) + 2 / x i = 2 * x i + 1 / x (i + 1)) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1996 → ℝ, 
    (∀ i, y i > 0) ∧ 
    (y 0 = y 1995) ∧ 
    (∀ i : Fin 1995, y (i + 1) + 2 / y i = 2 * y i + 1 / y (i + 1)) ∧
    y 0 = 2^997 :=
by sorry

end NUMINAMATH_CALUDE_max_x0_value_l4070_407097


namespace NUMINAMATH_CALUDE_mathematics_collections_l4070_407012

def word : String := "MATHEMATICS"

def num_vowels : Nat := 4
def num_consonants : Nat := 7
def num_ts : Nat := 2

def vowels_fall_off : Nat := 3
def consonants_fall_off : Nat := 4

def distinct_collections : Nat := 220

theorem mathematics_collections :
  (word.length = num_vowels + num_consonants) →
  (num_vowels = 4) →
  (num_consonants = 7) →
  (num_ts = 2) →
  (vowels_fall_off = 3) →
  (consonants_fall_off = 4) →
  distinct_collections = 220 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_collections_l4070_407012


namespace NUMINAMATH_CALUDE_machine_precision_test_l4070_407084

-- Define the sample data
def sample_data : List (Float × Nat) := [(3.0, 2), (3.5, 6), (3.8, 9), (4.4, 7), (4.5, 1)]

-- Define the hypothesized variance
def sigma_0_squared : Float := 0.1

-- Define the significance level
def alpha : Float := 0.05

-- Define the degrees of freedom
def df : Nat := 24

-- Function to calculate sample variance
def calculate_sample_variance (data : List (Float × Nat)) : Float :=
  sorry

-- Function to calculate chi-square test statistic
def calculate_chi_square (sample_variance : Float) (n : Nat) (sigma_0_squared : Float) : Float :=
  sorry

-- Function to get critical value from chi-square distribution
def get_chi_square_critical (alpha : Float) (df : Nat) : Float :=
  sorry

theorem machine_precision_test (data : List (Float × Nat)) (alpha : Float) (df : Nat) (sigma_0_squared : Float) :
  let sample_variance := calculate_sample_variance data
  let chi_square_obs := calculate_chi_square sample_variance data.length sigma_0_squared
  let chi_square_crit := get_chi_square_critical alpha df
  chi_square_obs > chi_square_crit :=
by
  sorry

#check machine_precision_test sample_data alpha df sigma_0_squared

end NUMINAMATH_CALUDE_machine_precision_test_l4070_407084


namespace NUMINAMATH_CALUDE_expected_girls_left_ten_boys_seven_girls_l4070_407030

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys : ℕ) (num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_ten_boys_seven_girls :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_left_ten_boys_seven_girls_l4070_407030


namespace NUMINAMATH_CALUDE_pencil_distribution_l4070_407075

theorem pencil_distribution (P : ℕ) (h : P % 9 = 8) :
  ∃ k : ℕ, P = 9 * k + 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4070_407075


namespace NUMINAMATH_CALUDE_fraction_equality_l4070_407046

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 * x + y) / (x - 3 * y) = -2) : 
  (x + 3 * y) / (3 * x - y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4070_407046


namespace NUMINAMATH_CALUDE_binomial_5_choose_3_l4070_407029

theorem binomial_5_choose_3 : Nat.choose 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_5_choose_3_l4070_407029


namespace NUMINAMATH_CALUDE_methane_moles_needed_l4070_407073

/-- Represents the chemical reaction C6H6 + CH4 → C7H8 + H2 -/
structure ChemicalReaction where
  benzene : ℝ
  methane : ℝ
  toluene : ℝ
  hydrogen : ℝ

/-- The molar mass of Benzene in g/mol -/
def benzene_molar_mass : ℝ := 78

/-- The total amount of Benzene required in grams -/
def total_benzene : ℝ := 156

/-- The number of moles of Toluene produced -/
def toluene_moles : ℝ := 2

/-- The number of moles of Hydrogen produced -/
def hydrogen_moles : ℝ := 2

theorem methane_moles_needed (reaction : ChemicalReaction) :
  reaction.benzene = total_benzene / benzene_molar_mass ∧
  reaction.toluene = toluene_moles ∧
  reaction.hydrogen = hydrogen_moles ∧
  reaction.benzene = reaction.methane →
  reaction.methane = 2 := by
  sorry

end NUMINAMATH_CALUDE_methane_moles_needed_l4070_407073


namespace NUMINAMATH_CALUDE_soldier_arrangement_l4070_407015

theorem soldier_arrangement (x : ℕ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 5 = 3) :
  x % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_soldier_arrangement_l4070_407015


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l4070_407007

/-- Represents a 9x9 grid filled with numbers 1 to 81 in row-major order -/
def Grid := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) in the grid -/
def gridValue (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The grid filled with numbers 1 to 81 -/
def numberGrid : Grid :=
  λ i j => gridValue i j

/-- The sum of the numbers in the four corners of the grid -/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

theorem corner_sum_is_164 :
  cornerSum numberGrid = 164 := by sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l4070_407007


namespace NUMINAMATH_CALUDE_line_through_points_circle_through_points_circle_center_on_y_axis_l4070_407039

-- Define the points A and B
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 3)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + (y-2)^2 = 2

-- Theorem for the line equation
theorem line_through_points : 
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 := by sorry

-- Theorem for the circle equation
theorem circle_through_points : 
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 := by sorry

-- Theorem to prove the center of the circle is on the y-axis
theorem circle_center_on_y_axis : 
  ∃ y : ℝ, ∀ x : ℝ, circle_eq 0 y → circle_eq x y → x = 0 := by sorry

end NUMINAMATH_CALUDE_line_through_points_circle_through_points_circle_center_on_y_axis_l4070_407039


namespace NUMINAMATH_CALUDE_min_omega_for_coinciding_symmetry_axes_l4070_407005

/-- Given a sinusoidal function y = 2sin(ωx + π/3) where ω > 0, 
    if the graph is shifted left and right by π/3 units and 
    the axes of symmetry of the resulting graphs coincide, 
    then the minimum value of ω is 3/2. -/
theorem min_omega_for_coinciding_symmetry_axes (ω : ℝ) : 
  ω > 0 → 
  (∀ x : ℝ, ∃ y : ℝ, y = 2 * Real.sin (ω * x + π / 3)) →
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = 2 * Real.sin (ω * (x + π / 3) + π / 3) ∧
    y₂ = 2 * Real.sin (ω * (x - π / 3) + π / 3)) →
  (∃ k : ℤ, ω * (π / 3) = k * π) →
  ω ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_coinciding_symmetry_axes_l4070_407005


namespace NUMINAMATH_CALUDE_fourth_person_height_l4070_407019

/-- Represents the heights of four people standing in order of increasing height. -/
structure HeightGroup where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the height problem. -/
def height_conditions (h : HeightGroup) : Prop :=
  h.second = h.first + 2 ∧
  h.third = h.second + 2 ∧
  h.fourth = h.third + 6 ∧
  (h.first + h.second + h.third + h.fourth) / 4 = 79

/-- The theorem stating that under the given conditions, the fourth person is 85 inches tall. -/
theorem fourth_person_height (h : HeightGroup) :
  height_conditions h → h.fourth = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l4070_407019


namespace NUMINAMATH_CALUDE_time_after_minutes_l4070_407038

def minutes_after_midnight : ℕ := 2345

def hours_in_day : ℕ := 24

def minutes_in_hour : ℕ := 60

def start_date : String := "January 1, 2022"

theorem time_after_minutes (m : ℕ) (h : m = minutes_after_midnight) :
  (start_date, m) = ("January 2", 15 * minutes_in_hour + 5) := by sorry

end NUMINAMATH_CALUDE_time_after_minutes_l4070_407038


namespace NUMINAMATH_CALUDE_rectangle_area_change_l4070_407080

/-- Proves that when a rectangle's length is increased by 15% and its breadth is decreased by 20%, 
    the resulting area is 92% of the original area. -/
theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (L * 1.15) * (B * 0.8) = 0.92 * (L * B) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l4070_407080


namespace NUMINAMATH_CALUDE_complex_modulus_power_eight_l4070_407036

theorem complex_modulus_power_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_eight_l4070_407036


namespace NUMINAMATH_CALUDE_rotational_inertia_scaling_l4070_407065

/-- Represents a sphere with a given radius and rotational inertia about its center axis -/
structure Sphere where
  radius : ℝ
  rotationalInertia : ℝ

/-- Given two spheres with the same density, where the second sphere has twice the radius of the first,
    prove that the rotational inertia of the second sphere is 32 times that of the first sphere -/
theorem rotational_inertia_scaling (s1 s2 : Sphere) (h1 : s2.radius = 2 * s1.radius) :
  s2.rotationalInertia = 32 * s1.rotationalInertia := by
  sorry


end NUMINAMATH_CALUDE_rotational_inertia_scaling_l4070_407065


namespace NUMINAMATH_CALUDE_literature_study_time_l4070_407024

def science_time : ℕ := 60
def math_time : ℕ := 80
def total_time_hours : ℕ := 3

theorem literature_study_time :
  total_time_hours * 60 - science_time - math_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_literature_study_time_l4070_407024


namespace NUMINAMATH_CALUDE_divisible_by_five_l4070_407047

theorem divisible_by_five (B : ℕ) : 
  B < 10 → (5270 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by sorry

end NUMINAMATH_CALUDE_divisible_by_five_l4070_407047


namespace NUMINAMATH_CALUDE_min_saplings_needed_l4070_407018

theorem min_saplings_needed (road_length : ℕ) (tree_spacing : ℕ) : road_length = 1000 → tree_spacing = 100 → 
  (road_length / tree_spacing + 1) * 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_saplings_needed_l4070_407018


namespace NUMINAMATH_CALUDE_tree_planting_activity_l4070_407017

theorem tree_planting_activity (boys girls : ℕ) : 
  (boys = 2 * girls + 15) →
  (girls = boys / 3 + 6) →
  (boys = 81 ∧ girls = 33) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_activity_l4070_407017


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l4070_407086

theorem largest_n_with_unique_k : 
  (∀ n : ℕ+, n > 1 → 
    ¬(∃! k : ℤ, (3 : ℚ)/7 < (n : ℚ)/((n : ℚ) + k) ∧ (n : ℚ)/((n : ℚ) + k) < 8/19)) ∧
  (∃! k : ℤ, (3 : ℚ)/7 < (1 : ℚ)/((1 : ℚ) + k) ∧ (1 : ℚ)/((1 : ℚ) + k) < 8/19) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l4070_407086


namespace NUMINAMATH_CALUDE_total_cost_is_36_l4070_407052

-- Define the cost per dose for each antibiotic
def cost_a : ℚ := 3
def cost_b : ℚ := 4.5

-- Define the number of doses per week for each antibiotic
def doses_a : ℕ := 3 * 2  -- 3 days, twice a day
def doses_b : ℕ := 4 * 1  -- 4 days, once a day

-- Define the discount rate and the number of doses required for the discount
def discount_rate : ℚ := 0.2
def discount_doses : ℕ := 10

-- Define the total cost function
def total_cost : ℚ :=
  min (doses_a * cost_a) (discount_doses * cost_a * (1 - discount_rate)) +
  doses_b * cost_b

-- Theorem statement
theorem total_cost_is_36 : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_36_l4070_407052


namespace NUMINAMATH_CALUDE_marion_has_23_paperclips_l4070_407006

-- Define the variables
def x : ℚ := 30
def y : ℚ := 7

-- Define Yun's remaining paperclips
def yun_remaining : ℚ := 2/5 * x

-- Define Marion's paperclips
def marion_paperclips : ℚ := 4/3 * yun_remaining + y

-- Theorem to prove
theorem marion_has_23_paperclips : marion_paperclips = 23 := by
  sorry

end NUMINAMATH_CALUDE_marion_has_23_paperclips_l4070_407006


namespace NUMINAMATH_CALUDE_expression_simplification_l4070_407028

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (a - 1) / (a + 2) / ((a^2 - 2*a) / (a^2 - 4)) - (a + 1) / a = -2 / a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4070_407028


namespace NUMINAMATH_CALUDE_ellipse_intersection_sum_of_squares_l4070_407035

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  a : ℝ
  b : ℝ

def Ellipse.standard : Ellipse := { a := 2, b := 1 }

/-- Check if a point is on the ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a line intersects the ellipse -/
def Line.intersectsEllipse (l : Line) (e : Ellipse) : Prop :=
  ∃ p : Point, e.contains p ∧ p.y = l.slope * p.x + l.intercept

/-- Calculate the distance squared from origin to a point -/
def Point.distanceSquared (p : Point) : ℝ :=
  p.x^2 + p.y^2

/-- Theorem statement -/
theorem ellipse_intersection_sum_of_squares :
  ∀ (l : Line),
    l.slope = 1/2 ∨ l.slope = -1/2 →
    l.intercept ≠ 0 →
    l.intersectsEllipse Ellipse.standard →
    ∃ (p1 p2 : Point),
      Ellipse.standard.contains p1 ∧
      Ellipse.standard.contains p2 ∧
      p1 ≠ p2 ∧
      p1.y = l.slope * p1.x + l.intercept ∧
      p2.y = l.slope * p2.x + l.intercept ∧
      p1.distanceSquared + p2.distanceSquared = 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_sum_of_squares_l4070_407035


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_l4070_407092

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_l4070_407092


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l4070_407031

def f (x : ℝ) : ℝ := (4*x)^2 - (4*x) + 2

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 10 ∧ f z₂ = 10 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/16) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l4070_407031


namespace NUMINAMATH_CALUDE_sum_of_cubes_l4070_407070

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l4070_407070


namespace NUMINAMATH_CALUDE_minimum_ladder_rungs_l4070_407093

theorem minimum_ladder_rungs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let n := a + b - Nat.gcd a b
  ∀ m : ℕ, m < n → ¬ (∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = m) ∧
  ∃ (x y : ℤ), x ≥ 0 ∧ y ≥ 0 ∧ a * x - b * y = n :=
by sorry

end NUMINAMATH_CALUDE_minimum_ladder_rungs_l4070_407093


namespace NUMINAMATH_CALUDE_angle_bisector_of_lines_l4070_407089

-- Define the two lines
def L₁ (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def L₂ (x y : ℝ) : Prop := 12 * x + 5 * y + 13 = 0

-- Define the angle bisector
def angle_bisector (x y : ℝ) : Prop := 56 * x - 7 * y + 39 = 0

-- Theorem statement
theorem angle_bisector_of_lines :
  ∀ x y : ℝ, angle_bisector x y ↔ (L₁ x y ∧ L₂ x y → ∃ t : ℝ, t > 0 ∧ 
    abs ((4 * x - 3 * y + 1) / (12 * x + 5 * y + 13)) = t) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_of_lines_l4070_407089


namespace NUMINAMATH_CALUDE_complete_square_equation_l4070_407081

theorem complete_square_equation : ∃ (a b c : ℤ), a > 0 ∧ 
  (∀ x : ℝ, 100 * x^2 + 60 * x - 49 = 0 ↔ (a * x + b)^2 = c) ∧
  a = 10 ∧ b = 3 ∧ c = 58 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l4070_407081


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4070_407020

theorem geometric_sequence_sum (a₀ r : ℚ) (n : ℕ) (h₁ : a₀ = 1/3) (h₂ : r = 1/3) (h₃ : n = 10) :
  let S := a₀ * (1 - r^n) / (1 - r)
  S = 29524/59049 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4070_407020


namespace NUMINAMATH_CALUDE_system_reliability_l4070_407033

theorem system_reliability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 0.9) 
  (h_B : p_B = 0.8) 
  (h_C : p_C = 0.7) : 
  p_A * p_B * p_C = 0.504 := by
sorry

end NUMINAMATH_CALUDE_system_reliability_l4070_407033


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l4070_407085

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (1, 2) and b = (-1, m), if they are perpendicular, then m = 1/2 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  perpendicular a b → m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l4070_407085


namespace NUMINAMATH_CALUDE_fold_square_diagonal_l4070_407064

/-- Given a square ABCD with side length 8 cm, where corner C is folded to point E
    (located 1/3 of the way from A to D on AD), prove that the length of FD is 32/9 cm,
    where F is the point where the fold intersects CD. -/
theorem fold_square_diagonal (A B C D E F G : ℝ × ℝ) : 
  let side_length : ℝ := 8
  -- ABCD is a square
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = side_length ∧ B.2 = 0) →
  (C.1 = side_length ∧ C.2 = side_length) →
  (D.1 = 0 ∧ D.2 = side_length) →
  -- E is one-third of the way along AD
  (E.1 = 0 ∧ E.2 = side_length / 3) →
  -- F is on CD
  (F.1 = side_length ∧ F.2 ≥ 0 ∧ F.2 ≤ side_length) →
  -- C coincides with E after folding
  (dist C E = dist C F) →
  -- FD length
  dist F D = 32 / 9 := by
sorry

end NUMINAMATH_CALUDE_fold_square_diagonal_l4070_407064


namespace NUMINAMATH_CALUDE_puppies_per_dog_l4070_407076

theorem puppies_per_dog (num_dogs : ℕ) (sold_fraction : ℚ) (price_per_puppy : ℕ) (total_revenue : ℕ) :
  num_dogs = 2 →
  sold_fraction = 3 / 4 →
  price_per_puppy = 200 →
  total_revenue = 3000 →
  (total_revenue / price_per_puppy : ℚ) / sold_fraction / num_dogs = 10 :=
by sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l4070_407076


namespace NUMINAMATH_CALUDE_three_heads_in_four_tosses_l4070_407060

/-- The probability of getting exactly k successes in n trials with probability p for each trial -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- A fair coin has probability 0.5 of landing heads -/
def fairCoinProbability : ℝ := 0.5

theorem three_heads_in_four_tosses :
  binomialProbability 4 3 fairCoinProbability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_four_tosses_l4070_407060


namespace NUMINAMATH_CALUDE_original_order_cost_l4070_407048

def original_tomatoes : ℝ := 0.99
def new_tomatoes : ℝ := 2.20
def original_lettuce : ℝ := 1.00
def new_lettuce : ℝ := 1.75
def original_celery : ℝ := 1.96
def new_celery : ℝ := 2.00
def delivery_and_tip : ℝ := 8.00
def new_bill : ℝ := 35.00

theorem original_order_cost :
  let tomatoes_diff := new_tomatoes - original_tomatoes
  let lettuce_diff := new_lettuce - original_lettuce
  let celery_diff := new_celery - original_celery
  let total_diff := tomatoes_diff + lettuce_diff + celery_diff
  new_bill - delivery_and_tip - total_diff = 25 := by sorry

end NUMINAMATH_CALUDE_original_order_cost_l4070_407048


namespace NUMINAMATH_CALUDE_absolute_value_equation_l4070_407027

theorem absolute_value_equation (x : ℝ) : 
  |2*x - 1| = Real.sqrt 2 - 1 → x = Real.sqrt 2 / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l4070_407027


namespace NUMINAMATH_CALUDE_triangle_longest_side_l4070_407068

theorem triangle_longest_side (x y : ℝ) :
  10 + (2 * y + 3) + (3 * x + 2) = 45 →
  (10 > 0) ∧ (2 * y + 3 > 0) ∧ (3 * x + 2 > 0) →
  max 10 (max (2 * y + 3) (3 * x + 2)) ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l4070_407068


namespace NUMINAMATH_CALUDE_distance_between_points_l4070_407008

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3, 4)
  let p2 : ℝ × ℝ := (4, -5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4070_407008


namespace NUMINAMATH_CALUDE_alcohol_solution_volume_l4070_407014

/-- Given an initial solution with volume V and 5% alcohol concentration,
    adding 5.5 liters of alcohol and 4.5 liters of water results in
    a new solution with 15% alcohol concentration if and only if
    the initial volume V is 40 liters. -/
theorem alcohol_solution_volume (V : ℝ) : 
  (0.15 * (V + 10) = 0.05 * V + 5.5) ↔ V = 40 := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_volume_l4070_407014


namespace NUMINAMATH_CALUDE_intersection_line_is_canonical_l4070_407042

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 2*x + 3*y + z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3*y - 2*z + 3 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 3)/(-3) = y/5 ∧ y/5 = z/(-9)

-- Theorem statement
theorem intersection_line_is_canonical :
  ∀ x y z : ℝ, plane1 x y z → plane2 x y z → canonical_line x y z :=
sorry

end NUMINAMATH_CALUDE_intersection_line_is_canonical_l4070_407042


namespace NUMINAMATH_CALUDE_green_chips_count_l4070_407087

theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 →
  blue = total / 10 →
  white = total / 2 →
  green = total - blue - white →
  green = 12 := by
  sorry

end NUMINAMATH_CALUDE_green_chips_count_l4070_407087


namespace NUMINAMATH_CALUDE_unique_root_of_cubic_l4070_407040

/-- The function f(x) = (x-3)(x^2+2x+3) has exactly one real root. -/
theorem unique_root_of_cubic (x : ℝ) : ∃! a : ℝ, (a - 3) * (a^2 + 2*a + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_cubic_l4070_407040


namespace NUMINAMATH_CALUDE_geometric_progression_a5_l4070_407022

-- Define a geometric progression
def isGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_progression_a5 (a : ℕ → ℝ) :
  isGeometricProgression a →
  (a 3) ^ 2 - 5 * (a 3) + 4 = 0 →
  (a 7) ^ 2 - 5 * (a 7) + 4 = 0 →
  a 5 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_a5_l4070_407022


namespace NUMINAMATH_CALUDE_two_solutions_for_equation_l4070_407037

theorem two_solutions_for_equation : 
  ∃! (n : ℕ), n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ (a + b + 3)^2 = 4*(a^2 + b^2))
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card 
  ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_two_solutions_for_equation_l4070_407037


namespace NUMINAMATH_CALUDE_unwashed_shirts_l4070_407099

theorem unwashed_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → washed = 29 → 
  short_sleeve + long_sleeve - washed = 1 := by
sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l4070_407099


namespace NUMINAMATH_CALUDE_probability_intersection_independent_events_l4070_407013

theorem probability_intersection_independent_events 
  (a b : Set ℝ) 
  (p : Set ℝ → ℝ) 
  (h1 : p a = 5/7) 
  (h2 : p b = 2/5) 
  (h3 : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_probability_intersection_independent_events_l4070_407013


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l4070_407077

/-- A hyperbola with given asymptote and foci x-coordinate -/
structure Hyperbola where
  asymptote : ℝ → ℝ
  foci_x : ℝ
  asymptote_eq : asymptote = fun x ↦ 2 * x + 3
  foci_x_eq : foci_x = 7

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 31

/-- Theorem stating that the other asymptote has the correct equation -/
theorem other_asymptote_equation (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 31 := by
  sorry


end NUMINAMATH_CALUDE_other_asymptote_equation_l4070_407077


namespace NUMINAMATH_CALUDE_number_line_is_line_l4070_407023

/-- A number line represents the set of real numbers. -/
def NumberLine : Type := ℝ

/-- A line is an infinite one-dimensional figure extending in both directions. -/
def Line : Type := ℝ

/-- A number line is equivalent to a line. -/
theorem number_line_is_line : NumberLine ≃ Line := by sorry

end NUMINAMATH_CALUDE_number_line_is_line_l4070_407023


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4070_407088

theorem triangle_angle_measure (A B C : ℝ) : 
  -- ABC is a triangle (sum of angles is 180°)
  A + B + C = 180 →
  -- Measure of angle C is 3/2 times the measure of angle B
  C = (3/2) * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 105°
  A = 105 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4070_407088


namespace NUMINAMATH_CALUDE_power_of_product_l4070_407053

theorem power_of_product (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4070_407053


namespace NUMINAMATH_CALUDE_point_on_exponential_graph_l4070_407096

theorem point_on_exponential_graph (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∃ P : ℝ × ℝ, ∀ x : ℝ, a^(x + 2) = P.2 → x = P.1 → P = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_exponential_graph_l4070_407096


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l4070_407066

theorem quadratic_roots_properties (a b : ℝ) : 
  (a^2 + 3*a - 2 = 0) → (b^2 + 3*b - 2 = 0) → 
  (a + b = -3) ∧ (a^3 + 3*a^2 + 2*b = -6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l4070_407066


namespace NUMINAMATH_CALUDE_prob_at_most_one_girl_l4070_407032

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of students in the group -/
def total_students : ℕ := num_boys + num_girls

/-- The number of students to be selected -/
def selected_students : ℕ := 2

/-- The probability of selecting at most one girl when randomly choosing 2 students from the group -/
theorem prob_at_most_one_girl : 
  (Nat.choose total_students selected_students - Nat.choose num_girls selected_students) / 
  Nat.choose total_students selected_students = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_girl_l4070_407032


namespace NUMINAMATH_CALUDE_niklaus_distance_l4070_407078

theorem niklaus_distance (lionel_miles : ℕ) (esther_yards : ℕ) (total_feet : ℕ) :
  lionel_miles = 4 →
  esther_yards = 975 →
  total_feet = 25332 →
  ∃ niklaus_feet : ℕ,
    niklaus_feet = total_feet - (lionel_miles * 5280 + esther_yards * 3) ∧
    niklaus_feet = 1287 :=
by sorry

end NUMINAMATH_CALUDE_niklaus_distance_l4070_407078


namespace NUMINAMATH_CALUDE_no_consecutive_beeches_probability_l4070_407045

/-- The number of oaks to be planted -/
def num_oaks : ℕ := 3

/-- The number of holm oaks to be planted -/
def num_holm_oaks : ℕ := 4

/-- The number of beeches to be planted -/
def num_beeches : ℕ := 5

/-- The total number of trees to be planted -/
def total_trees : ℕ := num_oaks + num_holm_oaks + num_beeches

/-- The probability of no two beeches being consecutive when planted randomly -/
def prob_no_consecutive_beeches : ℚ := 7 / 99

theorem no_consecutive_beeches_probability :
  let total_arrangements := (total_trees.factorial) / (num_oaks.factorial * num_holm_oaks.factorial * num_beeches.factorial)
  let favorable_arrangements := (Nat.choose 8 5) * ((num_oaks + num_holm_oaks).factorial / (num_oaks.factorial * num_holm_oaks.factorial))
  (favorable_arrangements : ℚ) / total_arrangements = prob_no_consecutive_beeches := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_beeches_probability_l4070_407045


namespace NUMINAMATH_CALUDE_cd_purchase_remaining_money_l4070_407011

theorem cd_purchase_remaining_money (total_money : ℚ) (num_cds : ℕ) (cd_price : ℚ) :
  (total_money / 5 = num_cds / 3 * cd_price) →
  (total_money - num_cds * cd_price) / total_money = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_remaining_money_l4070_407011


namespace NUMINAMATH_CALUDE_math_contest_grade11_score_l4070_407009

theorem math_contest_grade11_score (n : ℕ) (grade11_score : ℝ) :
  let grade11_count : ℝ := 0.2 * n
  let grade12_count : ℝ := 0.8 * n
  let overall_average : ℝ := 78
  let grade12_average : ℝ := 75
  (grade11_count * grade11_score + grade12_count * grade12_average) / n = overall_average →
  grade11_score = 90 := by
sorry

end NUMINAMATH_CALUDE_math_contest_grade11_score_l4070_407009


namespace NUMINAMATH_CALUDE_largest_multiple_in_sequence_l4070_407003

theorem largest_multiple_in_sequence : 
  ∀ (n : ℕ), 
  (3*n + 3*(n+1) + 3*(n+2) = 117) → 
  (max (3*n) (max (3*(n+1)) (3*(n+2))) = 42) := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_in_sequence_l4070_407003


namespace NUMINAMATH_CALUDE_student_distribution_count_l4070_407051

/-- The number of ways to distribute 5 students into three groups -/
def distribute_students : ℕ :=
  -- The actual distribution logic would go here
  80

/-- The conditions for the distribution -/
def valid_distribution (a b c : ℕ) : Prop :=
  a + b + c = 5 ∧ a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 1

theorem student_distribution_count :
  ∃ (a b c : ℕ), valid_distribution a b c ∧
  (∀ (x y z : ℕ), valid_distribution x y z → x + y + z = 5) ∧
  distribute_students = 80 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_count_l4070_407051


namespace NUMINAMATH_CALUDE_discount_clinic_visits_prove_discount_clinic_visits_l4070_407050

def normal_doctor_charge : ℝ := 200
def discount_percentage : ℝ := 0.7
def savings : ℝ := 80

theorem discount_clinic_visits : ℝ :=
  let discount_clinic_charge := normal_doctor_charge * (1 - discount_percentage)
  let total_paid := normal_doctor_charge - savings
  total_paid / discount_clinic_charge

theorem prove_discount_clinic_visits :
  discount_clinic_visits = 2 := by sorry

end NUMINAMATH_CALUDE_discount_clinic_visits_prove_discount_clinic_visits_l4070_407050


namespace NUMINAMATH_CALUDE_first_shift_participation_is_twenty_percent_l4070_407090

/-- Represents a company with three shifts of employees and a pension program. -/
structure Company where
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  second_shift_participation : ℚ
  third_shift_participation : ℚ
  total_participation : ℚ

/-- The percentage of first shift employees participating in the pension program. -/
def first_shift_participation (c : Company) : ℚ :=
  let total_employees := c.first_shift + c.second_shift + c.third_shift
  let total_participants := (c.total_participation * total_employees) / 100
  let second_shift_participants := (c.second_shift_participation * c.second_shift) / 100
  let third_shift_participants := (c.third_shift_participation * c.third_shift) / 100
  let first_shift_participants := total_participants - second_shift_participants - third_shift_participants
  (first_shift_participants * 100) / c.first_shift

theorem first_shift_participation_is_twenty_percent (c : Company) 
  (h1 : c.first_shift = 60)
  (h2 : c.second_shift = 50)
  (h3 : c.third_shift = 40)
  (h4 : c.second_shift_participation = 40)
  (h5 : c.third_shift_participation = 10)
  (h6 : c.total_participation = 24) :
  first_shift_participation c = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_shift_participation_is_twenty_percent_l4070_407090


namespace NUMINAMATH_CALUDE_votes_difference_l4070_407082

theorem votes_difference (total_votes : ℕ) (against_percentage : ℚ) : 
  total_votes = 330 →
  against_percentage = 40 / 100 →
  (total_votes : ℚ) * (1 - against_percentage) - (total_votes : ℚ) * against_percentage = 66 := by
  sorry

end NUMINAMATH_CALUDE_votes_difference_l4070_407082


namespace NUMINAMATH_CALUDE_simplify_fraction_l4070_407034

theorem simplify_fraction (a : ℝ) : 
  (1 + a^2 / (1 + 2*a)) / ((1 + a) / (1 + 2*a)) = 1 + a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4070_407034


namespace NUMINAMATH_CALUDE_contributions_before_johns_l4070_407063

def average_before_johns (n : ℕ) : ℚ := 50

def johns_contribution : ℚ := 150

def new_average (n : ℕ) : ℚ := 75

def total_before_johns (n : ℕ) : ℚ := n * average_before_johns n

def total_after_johns (n : ℕ) : ℚ := total_before_johns n + johns_contribution

theorem contributions_before_johns :
  ∃ n : ℕ, 
    (new_average n = (3/2) * average_before_johns n) ∧
    (new_average n = 75) ∧
    (johns_contribution = 150) ∧
    (new_average n = total_after_johns n / (n + 1)) ∧
    (n = 3) :=
by sorry

end NUMINAMATH_CALUDE_contributions_before_johns_l4070_407063


namespace NUMINAMATH_CALUDE_vehicles_with_high_speed_l4070_407094

theorem vehicles_with_high_speed (vehicles_80_to_89 vehicles_90_to_99 vehicles_100_to_109 : ℕ) :
  vehicles_80_to_89 = 15 →
  vehicles_90_to_99 = 30 →
  vehicles_100_to_109 = 5 →
  vehicles_80_to_89 + vehicles_90_to_99 + vehicles_100_to_109 = 50 :=
by sorry

end NUMINAMATH_CALUDE_vehicles_with_high_speed_l4070_407094


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l4070_407025

/-- Given two lines that intersect at x = -12, prove that k = 65 -/
theorem intersection_point_k_value :
  ∀ (y : ℝ),
  -3 * (-12) + y = k →
  0.75 * (-12) + y = 20 →
  k = 65 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l4070_407025


namespace NUMINAMATH_CALUDE_zoe_winter_clothing_boxes_l4070_407054

theorem zoe_winter_clothing_boxes :
  let items_per_box := 4 + 6  -- 4 scarves and 6 mittens per box
  let total_items := 80       -- total pieces of winter clothing
  total_items / items_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_zoe_winter_clothing_boxes_l4070_407054


namespace NUMINAMATH_CALUDE_cat_food_insufficiency_l4070_407079

theorem cat_food_insufficiency (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end NUMINAMATH_CALUDE_cat_food_insufficiency_l4070_407079


namespace NUMINAMATH_CALUDE_quadrilateral_area_l4070_407021

/-- The area of a quadrilateral with a diagonal of length 40 and offsets 11 and 9 is 400 -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) : 
  diagonal = 40 → offset1 = 11 → offset2 = 9 → 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 400 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l4070_407021


namespace NUMINAMATH_CALUDE_analytical_method_seeks_sufficient_conditions_l4070_407061

/-- The analytical method for proving inequalities -/
structure AnalyticalMethod where
  /-- The method proceeds from effect to cause -/
  effect_to_cause : Bool

/-- A condition in the context of proving inequalities -/
inductive Condition
  | Necessary
  | Sufficient
  | NecessaryAndSufficient
  | NecessaryOrSufficient

/-- The reasoning process sought by the analytical method -/
def reasoning_process (method : AnalyticalMethod) : Condition :=
  Condition.Sufficient

/-- Theorem stating that the analytical method seeks sufficient conditions -/
theorem analytical_method_seeks_sufficient_conditions (method : AnalyticalMethod) :
  reasoning_process method = Condition.Sufficient := by
  sorry

end NUMINAMATH_CALUDE_analytical_method_seeks_sufficient_conditions_l4070_407061


namespace NUMINAMATH_CALUDE_basketball_reach_theorem_l4070_407026

/-- Represents the height a basketball player can reach above their head using their arms -/
def reachAboveHead (playerHeight rimHeight jumpHeight : ℕ) : ℕ :=
  rimHeight * 12 + 6 - (playerHeight * 12 + jumpHeight)

/-- Theorem stating that a 6-foot tall player who can jump 32 inches high needs to reach 22 inches above their head to dunk on a 10-foot rim -/
theorem basketball_reach_theorem :
  reachAboveHead 6 10 32 = 22 := by
  sorry

end NUMINAMATH_CALUDE_basketball_reach_theorem_l4070_407026


namespace NUMINAMATH_CALUDE_fifth_dog_weight_l4070_407056

theorem fifth_dog_weight (w1 w2 w3 w4 y : ℝ) : 
  w1 = 25 ∧ w2 = 31 ∧ w3 = 35 ∧ w4 = 33 →
  (w1 + w2 + w3 + w4) / 4 = (w1 + w2 + w3 + w4 + y) / 5 →
  y = 31 :=
by sorry

end NUMINAMATH_CALUDE_fifth_dog_weight_l4070_407056


namespace NUMINAMATH_CALUDE_all_descendants_have_no_daughters_l4070_407049

/-- Represents Bertha's family tree -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  great_granddaughters : ℕ

/-- The number of Bertha's daughters who have daughters -/
def daughters_with_daughters (f : BerthaFamily) : ℕ := f.granddaughters / 5

/-- The number of Bertha's descendants who have no daughters -/
def descendants_without_daughters (f : BerthaFamily) : ℕ :=
  f.daughters + f.granddaughters

theorem all_descendants_have_no_daughters (f : BerthaFamily) :
  f.daughters = 8 →
  f.daughters + f.granddaughters + f.great_granddaughters = 48 →
  f.great_granddaughters = 0 →
  daughters_with_daughters f * 5 = f.granddaughters →
  descendants_without_daughters f = f.daughters + f.granddaughters + f.great_granddaughters :=
by sorry

end NUMINAMATH_CALUDE_all_descendants_have_no_daughters_l4070_407049


namespace NUMINAMATH_CALUDE_sequence_general_term_l4070_407091

-- Define the sequence and its partial sum
def S (n : ℕ) : ℤ := n^2 - 4*n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 2*n - 5

-- Theorem statement
theorem sequence_general_term (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l4070_407091


namespace NUMINAMATH_CALUDE_secret_ballot_best_for_new_member_l4070_407043

/-- Represents a voting method -/
inductive VotingMethod
  | ShowOfHandsAgree
  | ShowOfHandsDisagree
  | SecretBallot
  | RecordedVote

/-- Represents the context of the vote -/
structure VoteContext where
  purpose : String

/-- Defines what it means for a voting method to reflect the true will of students -/
def reflectsTrueWill (method : VotingMethod) (context : VoteContext) : Prop := sorry

/-- Theorem stating that secret ballot best reflects the true will of students for adding a new class committee member -/
theorem secret_ballot_best_for_new_member :
  ∀ (context : VoteContext),
  context.purpose = "adding a new class committee member" →
  ∀ (method : VotingMethod),
  reflectsTrueWill VotingMethod.SecretBallot context →
  reflectsTrueWill method context →
  method = VotingMethod.SecretBallot :=
sorry

end NUMINAMATH_CALUDE_secret_ballot_best_for_new_member_l4070_407043


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l4070_407016

/-- Given a set of worksheets with some graded and some problems left to grade,
    calculate the number of problems per worksheet. -/
theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (problems_left : ℕ)
  (h1 : total_worksheets = 14)
  (h2 : graded_worksheets = 7)
  (h3 : problems_left = 14)
  (h4 : graded_worksheets < total_worksheets) :
  problems_left / (total_worksheets - graded_worksheets) = 2 :=
by
  sorry

#check problems_per_worksheet

end NUMINAMATH_CALUDE_problems_per_worksheet_l4070_407016


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l4070_407074

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l4070_407074


namespace NUMINAMATH_CALUDE_locus_of_M_l4070_407010

-- Define the constant k
variable (k : ℝ)

-- Define the coordinates of points A and B
variable (xA yA xB yB : ℝ)

-- Define the coordinates of point M
variable (xM yM : ℝ)

-- Axioms based on the problem conditions
axiom perpendicular_axes : xA * xB + yA * yB = 0
axiom A_on_x_axis : yA = 0
axiom B_on_y_axis : xB = 0
axiom sum_of_distances : Real.sqrt (xA^2 + yA^2) + Real.sqrt (xB^2 + yB^2) = k
axiom M_on_circumcircle : (xM - xA)^2 + (yM - yA)^2 = (xM - xB)^2 + (yM - yB)^2

-- Theorem statement
theorem locus_of_M : 
  (xM - k/2)^2 + (yM - k/2)^2 = k^2/2 := by sorry

end NUMINAMATH_CALUDE_locus_of_M_l4070_407010


namespace NUMINAMATH_CALUDE_coefficient_d_value_l4070_407055

-- Define the polynomial Q(x)
def Q (x d : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + d*x + 15

-- State the theorem
theorem coefficient_d_value :
  ∃ d : ℝ, (∀ x : ℝ, Q x d = 0 → x = -3) ∧ d = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_d_value_l4070_407055


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l4070_407069

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l4070_407069


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l4070_407059

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l4070_407059


namespace NUMINAMATH_CALUDE_initial_pens_l4070_407004

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 75 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 25 :=
by sorry

end NUMINAMATH_CALUDE_initial_pens_l4070_407004


namespace NUMINAMATH_CALUDE_sqrt_inequality_l4070_407072

theorem sqrt_inequality (a : ℝ) (h : a > 6) :
  Real.sqrt (a - 3) - Real.sqrt (a - 4) < Real.sqrt (a - 5) - Real.sqrt (a - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l4070_407072


namespace NUMINAMATH_CALUDE_round_trip_percentage_l4070_407057

/-- Represents the percentage of ship passengers -/
@[ext] structure ShipPassengers where
  total : ℝ
  roundTrip : ℝ
  roundTripWithCar : ℝ

/-- Conditions for the ship passengers -/
def validShipPassengers (p : ShipPassengers) : Prop :=
  0 ≤ p.roundTrip ∧ p.roundTrip ≤ 100 ∧
  0 ≤ p.roundTripWithCar ∧ p.roundTripWithCar ≤ p.roundTrip ∧
  p.roundTripWithCar = 0.4 * p.roundTrip

theorem round_trip_percentage (p : ShipPassengers) 
  (h : validShipPassengers p) : 
  p.roundTrip = p.roundTripWithCar / 0.4 :=
sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l4070_407057


namespace NUMINAMATH_CALUDE_minimum_games_for_winning_percentage_l4070_407058

theorem minimum_games_for_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (3 + k : ℚ) / (4 + k) < 4/5) ∧ 
  (3 + N : ℚ) / (4 + N) ≥ 4/5 → 
  N = 1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_games_for_winning_percentage_l4070_407058


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4070_407062

/-- Given an equilateral triangle with perimeter 45 and an isosceles triangle sharing one side
    with the equilateral triangle and having a base of length 10, the perimeter of the isosceles
    triangle is 40. -/
theorem isosceles_triangle_perimeter
  (equilateral_perimeter : ℝ)
  (isosceles_base : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 45)
  (h_isosceles_base : isosceles_base = 10)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   ∃ (leg : ℝ), leg = side) :
  ∃ (isosceles_perimeter : ℝ), isosceles_perimeter = 40 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4070_407062


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l4070_407071

/-- Given a geometric series {a_n} with positive terms, if a_3 = 18 and S_3 = 26, then q = 3 -/
theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_a3 : a 3 = 18)
  (h_S3 : S 3 = 26) :
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l4070_407071
