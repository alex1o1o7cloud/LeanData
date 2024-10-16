import Mathlib

namespace NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l911_91191

theorem log_43_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 43 / Real.log 10 ∧ Real.log 43 / Real.log 10 < b ∧ a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_43_between_consecutive_integers_l911_91191


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l911_91183

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0) → 
  (Real.cos (15 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) + Real.sin θ) → 
  θ = 15 * Real.pi / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l911_91183


namespace NUMINAMATH_CALUDE_competition_max_robot_weight_l911_91121

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The weight of the battery in pounds -/
def battery_weight : ℝ := 20

/-- The minimum additional weight above the standard robot in pounds -/
def min_additional_weight : ℝ := 5

/-- The maximum weight multiplier -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition, including the battery -/
def max_robot_weight : ℝ := 250

theorem competition_max_robot_weight :
  max_robot_weight = 
    max_weight_multiplier * (standard_robot_weight + min_additional_weight + battery_weight) :=
by sorry

end NUMINAMATH_CALUDE_competition_max_robot_weight_l911_91121


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l911_91152

theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₂ > x₁) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l911_91152


namespace NUMINAMATH_CALUDE_right_triangle_square_equal_area_l911_91167

theorem right_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) : 
  (1/2 * s * h = s^2) → h = 2*s := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_square_equal_area_l911_91167


namespace NUMINAMATH_CALUDE_knights_and_liars_l911_91157

/-- Represents the two types of inhabitants in the country -/
inductive Inhabitant
  | Knight
  | Liar

/-- The statement made by A -/
def statement (a b : Inhabitant) : Prop :=
  a = Inhabitant.Liar ∧ b ≠ Inhabitant.Liar

/-- A function that determines if a given statement is true based on the speaker's type -/
def isTrueStatement (speaker : Inhabitant) (stmt : Prop) : Prop :=
  (speaker = Inhabitant.Knight ∧ stmt) ∨ (speaker = Inhabitant.Liar ∧ ¬stmt)

theorem knights_and_liars (a b : Inhabitant) :
  isTrueStatement a (statement a b) →
  a = Inhabitant.Liar ∧ b = Inhabitant.Liar :=
by sorry

end NUMINAMATH_CALUDE_knights_and_liars_l911_91157


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_expression_l911_91133

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (k : ℕ) : ℚ :=
  let n : ℕ := 2 * k - 1
  let a₁ : ℚ := k^2 - 1
  let d : ℚ := 1
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the sum of the arithmetic sequence equals the given expression -/
theorem arithmetic_sum_equals_expression (k : ℕ) :
  arithmetic_sum k = 2 * k^3 + k^2 - 4 * k + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_expression_l911_91133


namespace NUMINAMATH_CALUDE_negation_equivalence_l911_91107

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l911_91107


namespace NUMINAMATH_CALUDE_derivative_x_plus_one_squared_times_x_minus_one_l911_91185

theorem derivative_x_plus_one_squared_times_x_minus_one (x : ℝ) :
  deriv (λ x => (x + 1)^2 * (x - 1)) x = 3*x^2 + 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_plus_one_squared_times_x_minus_one_l911_91185


namespace NUMINAMATH_CALUDE_perpendicular_lines_l911_91103

def line1 (x y : ℝ) : Prop := 3 * x - y = 6

def line2 (x y : ℝ) : Prop := y = -1/3 * x + 7/3

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ m1 m2 : ℝ, (∀ x y, f x y ↔ y = m1 * x + 0) ∧ 
              (∀ x y, g x y ↔ y = m2 * x + 0) ∧
              m1 * m2 = -1

theorem perpendicular_lines :
  perpendicular line1 line2 ∧ line2 (-2) 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l911_91103


namespace NUMINAMATH_CALUDE_factory_output_increase_l911_91173

/-- Proves that the percentage increase in actual output compared to last year is 11.1% -/
theorem factory_output_increase (a : ℝ) : 
  let last_year_output := a / 1.1
  let this_year_actual := a * 1.01
  (this_year_actual - last_year_output) / last_year_output * 100 = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_increase_l911_91173


namespace NUMINAMATH_CALUDE_bicycle_stand_stability_l911_91111

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- A bicycle stand is a device that supports a bicycle. -/
structure BicycleStand where
  -- We don't need to define the specifics of a bicycle stand for this problem

/-- Stability is a property that allows an object to remain balanced and resist toppling. -/
def Stability : Prop := sorry

/-- A property that allows an object to stand firmly on the ground. -/
def AllowsToStandFirmly (prop : Prop) : Prop := sorry

theorem bicycle_stand_stability (t : Triangle) (s : BicycleStand) :
  AllowsToStandFirmly Stability := by sorry

end NUMINAMATH_CALUDE_bicycle_stand_stability_l911_91111


namespace NUMINAMATH_CALUDE_missy_additional_capacity_l911_91158

/-- Proves that Missy can handle 15 more claims than John given the conditions -/
theorem missy_additional_capacity (jan_capacity : ℕ) (john_capacity : ℕ) (missy_capacity : ℕ) :
  jan_capacity = 20 →
  john_capacity = jan_capacity + (jan_capacity * 3 / 10) →
  missy_capacity = 41 →
  missy_capacity - john_capacity = 15 := by
  sorry

#check missy_additional_capacity

end NUMINAMATH_CALUDE_missy_additional_capacity_l911_91158


namespace NUMINAMATH_CALUDE_range_of_a_l911_91104

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (B a ⊆ Aᶜ) ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l911_91104


namespace NUMINAMATH_CALUDE_inverse_f_at_4_l911_91120

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

def HasInverse (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem inverse_f_at_4 (f_inv : ℝ → ℝ) (h : HasInverse f f_inv) : f_inv 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_l911_91120


namespace NUMINAMATH_CALUDE_f_13_equals_214_l911_91130

/-- The function f defined as f(n) = n^2 + 2n + 19 -/
def f (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that f(13) equals 214 -/
theorem f_13_equals_214 : f 13 = 214 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_214_l911_91130


namespace NUMINAMATH_CALUDE_total_bookmark_sales_l911_91114

/-- Represents the sales of bookmarks over two days -/
structure BookmarkSales where
  /-- Number of bookmarks sold on the first day -/
  day1 : ℕ
  /-- Number of bookmarks sold on the second day -/
  day2 : ℕ

/-- Theorem stating that the total number of bookmarks sold over two days is 3m-3 -/
theorem total_bookmark_sales (m : ℕ) (sales : BookmarkSales)
    (h1 : sales.day1 = m)
    (h2 : sales.day2 = 2 * m - 3) :
    sales.day1 + sales.day2 = 3 * m - 3 := by
  sorry

end NUMINAMATH_CALUDE_total_bookmark_sales_l911_91114


namespace NUMINAMATH_CALUDE_survey_respondents_l911_91125

theorem survey_respondents (brand_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  brand_x = 200 →
  ratio_x = 4 →
  ratio_y = 1 →
  ∃ total : ℕ, total = brand_x + (brand_x * ratio_y / ratio_x) ∧ total = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l911_91125


namespace NUMINAMATH_CALUDE_oldest_sibling_age_is_44_l911_91172

def kay_age : ℕ := 32

def youngest_sibling_age : ℕ := kay_age / 2 - 5

def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

theorem oldest_sibling_age_is_44 : oldest_sibling_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_is_44_l911_91172


namespace NUMINAMATH_CALUDE_shaded_area_is_24_l911_91181

structure Rectangle where
  width : ℝ
  height : ℝ

structure Triangle where
  base : ℝ
  height : ℝ

def shaded_area (rect : Rectangle) (tri : Triangle) : ℝ :=
  sorry

theorem shaded_area_is_24 (rect : Rectangle) (tri : Triangle) :
  rect.width = 8 ∧ rect.height = 12 ∧ tri.base = 8 ∧ tri.height = rect.height →
  shaded_area rect tri = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_24_l911_91181


namespace NUMINAMATH_CALUDE_brick_surface_area_l911_91116

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
sorry

end NUMINAMATH_CALUDE_brick_surface_area_l911_91116


namespace NUMINAMATH_CALUDE_largest_integer_solution_of_inequalities_l911_91140

theorem largest_integer_solution_of_inequalities :
  ∀ x : ℤ, (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) → x ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_of_inequalities_l911_91140


namespace NUMINAMATH_CALUDE_angle4_measure_l911_91160

-- Define the triangle and its angles
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)
  (angle5 : ℝ)
  (angle6 : ℝ)

-- Define the theorem
theorem angle4_measure (t : Triangle) 
  (h1 : t.angle1 = 76)
  (h2 : t.angle2 = 27)
  (h3 : t.angle3 = 17)
  (h4 : t.angle1 + t.angle2 + t.angle3 + t.angle5 + t.angle6 = 180) -- Sum of angles in the large triangle
  (h5 : t.angle4 + t.angle5 + t.angle6 = 180) -- Sum of angles in the small triangle
  : t.angle4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle4_measure_l911_91160


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l911_91144

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l911_91144


namespace NUMINAMATH_CALUDE_parallel_vector_scalar_l911_91169

/-- Given vectors a, b, and c in R², prove that if a + kb is parallel to c, then k = 1/2 -/
theorem parallel_vector_scalar (a b c : ℝ × ℝ) (k : ℝ) 
    (ha : a = (2, -1)) 
    (hb : b = (1, 1)) 
    (hc : c = (-5, 1)) 
    (h_parallel : ∃ (t : ℝ), a.1 + k * b.1 = t * c.1 ∧ a.2 + k * b.2 = t * c.2) : 
  k = 1/2 := by
  sorry

#check parallel_vector_scalar

end NUMINAMATH_CALUDE_parallel_vector_scalar_l911_91169


namespace NUMINAMATH_CALUDE_y_plus_z_value_l911_91194

theorem y_plus_z_value (x y z : ℕ) (hx : x = 4) (hy : y = 3 * x) (hz : z = 2 * y) : 
  y + z = 36 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_value_l911_91194


namespace NUMINAMATH_CALUDE_election_winner_votes_l911_91184

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (65 : ℚ) / 100 * total_votes - (35 : ℚ) / 100 * total_votes = 300) : 
  (65 : ℚ) / 100 * total_votes = 650 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l911_91184


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l911_91119

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
  | Hit
  | Miss

/-- Represents the outcome of two shots --/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once --/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times --/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

/-- Theorem stating that "missing the target both times" is the mutually exclusive event of "hitting the target at least once" --/
theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), hitAtLeastOnce outcome ↔ ¬(missBothTimes outcome) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l911_91119


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l911_91108

theorem purely_imaginary_complex (a : ℝ) : 
  (((2 : ℂ) + a * Complex.I) / ((1 : ℂ) - Complex.I) + (1 : ℂ) / ((1 : ℂ) + Complex.I)).re = 0 ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l911_91108


namespace NUMINAMATH_CALUDE_unique_x_value_l911_91153

theorem unique_x_value : ∃! x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 120) ∧ 
  (x < 25) ∧ 
  (x = 18) := by
sorry

end NUMINAMATH_CALUDE_unique_x_value_l911_91153


namespace NUMINAMATH_CALUDE_min_value_of_z_l911_91180

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := y - 2 * x

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is -7 -/
theorem min_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x' y' ≥ z x y ∧
  z x y = -7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l911_91180


namespace NUMINAMATH_CALUDE_claire_pets_l911_91161

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) 
  (h_total : total_pets = 92)
  (h_males : total_males = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry


end NUMINAMATH_CALUDE_claire_pets_l911_91161


namespace NUMINAMATH_CALUDE_position_from_front_l911_91190

theorem position_from_front (total : ℕ) (position_from_back : ℕ) (h1 : total = 22) (h2 : position_from_back = 13) :
  total - position_from_back + 1 = 10 := by
sorry

end NUMINAMATH_CALUDE_position_from_front_l911_91190


namespace NUMINAMATH_CALUDE_max_books_on_shelf_l911_91156

theorem max_books_on_shelf (n : ℕ) (s₁ s₂ S : ℕ) : 
  (S + s₁ ≥ (n - 2) / 2) →
  (S + s₂ < (n - 2) / 3) →
  (n ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_max_books_on_shelf_l911_91156


namespace NUMINAMATH_CALUDE_angle_sum_equality_l911_91139

theorem angle_sum_equality (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (eq1 : 4 * (Real.cos a)^2 + 3 * (Real.sin b)^2 = 1)
  (eq2 : 4 * Real.sin (2*a) + 3 * Real.cos (2*b) = 0) :
  a + 2*b = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l911_91139


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l911_91175

theorem sum_of_squares_and_products (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → a^2 + b^2 + c^2 = 39 → a*b + b*c + c*a = 21 → a + b + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l911_91175


namespace NUMINAMATH_CALUDE_integer_expression_is_integer_l911_91137

theorem integer_expression_is_integer (n : ℤ) : ∃ m : ℤ, (n / 3 + n^2 / 2 + n^3 / 6 : ℚ) = m := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_is_integer_l911_91137


namespace NUMINAMATH_CALUDE_quanxing_max_difference_l911_91166

/-- Represents the mass of a bottle of Quanxing mineral water in mL -/
structure QuanxingBottle where
  mass : ℝ
  h : abs (mass - 450) ≤ 1

/-- The maximum difference in mass between any two Quanxing bottles is 2 mL -/
theorem quanxing_max_difference (bottle1 bottle2 : QuanxingBottle) :
  abs (bottle1.mass - bottle2.mass) ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_quanxing_max_difference_l911_91166


namespace NUMINAMATH_CALUDE_chemical_B_calculation_l911_91197

/-- The amount of chemical B needed to create 1 liter of solution -/
def chemical_B_needed : ℚ :=
  2/3

/-- The amount of chemical B in the original mixture -/
def original_chemical_B : ℚ :=
  0.08

/-- The amount of water in the original mixture -/
def original_water : ℚ :=
  0.04

/-- The total amount of solution in the original mixture -/
def original_total : ℚ :=
  0.12

/-- The target amount of solution to be created -/
def target_amount : ℚ :=
  1

theorem chemical_B_calculation :
  original_chemical_B / original_total * target_amount = chemical_B_needed :=
by sorry

end NUMINAMATH_CALUDE_chemical_B_calculation_l911_91197


namespace NUMINAMATH_CALUDE_investment_total_calculation_l911_91141

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  amount1 : ℝ

/-- Calculates the total interest earned from an investment -/
def totalInterest (inv : Investment) : ℝ :=
  inv.rate1 * inv.amount1 + inv.rate2 * (inv.total - inv.amount1)

theorem investment_total_calculation (inv : Investment) 
  (h1 : inv.rate1 = 0.07)
  (h2 : inv.rate2 = 0.09)
  (h3 : inv.amount1 = 5500)
  (h4 : totalInterest inv = 970) :
  inv.total = 12000 := by
sorry

end NUMINAMATH_CALUDE_investment_total_calculation_l911_91141


namespace NUMINAMATH_CALUDE_trip_duration_l911_91117

/-- Proves that the trip duration is 24 hours given the specified conditions -/
theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 35 →
  initial_time = 4 →
  additional_speed = 53 →
  average_speed = 50 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_l911_91117


namespace NUMINAMATH_CALUDE_functional_equation_problem_l911_91129

/-- The functional equation problem -/
theorem functional_equation_problem :
  ∀ (f h : ℝ → ℝ),
  (∀ x y : ℝ, f (x^2 + y * h x) = x * h x + f (x * y)) →
  ((∃ a b : ℝ, (∀ x : ℝ, f x = a) ∧ 
                (∀ x : ℝ, x ≠ 0 → h x = 0) ∧ 
                (h 0 = b)) ∨
   (∃ a : ℝ, (∀ x : ℝ, f x = x + a) ∧ 
             (∀ x : ℝ, h x = x))) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l911_91129


namespace NUMINAMATH_CALUDE_velvet_for_cloak_l911_91179

/-- The number of hats that can be made from one yard of velvet -/
def hats_per_yard : ℕ := 4

/-- The total number of yards of velvet needed for 6 cloaks and 12 hats -/
def total_yards : ℕ := 21

/-- The number of cloaks made with the total yards -/
def num_cloaks : ℕ := 6

/-- The number of hats made with the total yards -/
def num_hats : ℕ := 12

/-- The number of yards needed to make one cloak -/
def yards_per_cloak : ℚ := 3

theorem velvet_for_cloak :
  yards_per_cloak = (total_yards - (num_hats / hats_per_yard : ℚ)) / num_cloaks := by
  sorry

end NUMINAMATH_CALUDE_velvet_for_cloak_l911_91179


namespace NUMINAMATH_CALUDE_negative_two_is_square_root_of_four_l911_91162

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem negative_two_is_square_root_of_four :
  is_square_root 4 (-2) :=
sorry

end NUMINAMATH_CALUDE_negative_two_is_square_root_of_four_l911_91162


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l911_91186

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 40 * S) : 
  (S - C) / C * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l911_91186


namespace NUMINAMATH_CALUDE_two_digit_congruent_to_two_mod_four_count_l911_91145

theorem two_digit_congruent_to_two_mod_four_count : 
  (Finset.filter (fun n => n ≥ 10 ∧ n ≤ 99 ∧ n % 4 = 2) (Finset.range 100)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_congruent_to_two_mod_four_count_l911_91145


namespace NUMINAMATH_CALUDE_max_points_without_equilateral_triangle_l911_91196

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the set of 10 points: vertices, centroid, and trisection points -/
def TrianglePoints (t : EquilateralTriangle) : Finset Point :=
  sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (p1 p2 p3 : Point) : Prop :=
  sorry

/-- The main theorem -/
theorem max_points_without_equilateral_triangle (t : EquilateralTriangle) :
  ∃ (s : Finset Point), s ⊆ TrianglePoints t ∧ s.card = 6 ∧
  ∀ (p1 p2 p3 : Point), p1 ∈ s → p2 ∈ s → p3 ∈ s → ¬(isEquilateral p1 p2 p3) ∧
  ∀ (s' : Finset Point), s' ⊆ TrianglePoints t →
    (∀ (p1 p2 p3 : Point), p1 ∈ s' → p2 ∈ s' → p3 ∈ s' → ¬(isEquilateral p1 p2 p3)) →
    s'.card ≤ 6 :=
  sorry

end NUMINAMATH_CALUDE_max_points_without_equilateral_triangle_l911_91196


namespace NUMINAMATH_CALUDE_postage_calculation_l911_91110

/-- Calculates the postage for a letter given the base fee, additional fee per ounce, and weight -/
def calculatePostage (baseFee : ℚ) (additionalFeePerOunce : ℚ) (weight : ℚ) : ℚ :=
  baseFee + additionalFeePerOunce * (weight - 1)

/-- Theorem stating that the postage for a 5.3 ounce letter is $1.425 under the given fee structure -/
theorem postage_calculation :
  let baseFee : ℚ := 35 / 100  -- 35 cents in dollars
  let additionalFeePerOunce : ℚ := 25 / 100  -- 25 cents in dollars
  let weight : ℚ := 53 / 10  -- 5.3 ounces
  calculatePostage baseFee additionalFeePerOunce weight = 1425 / 1000 := by
  sorry

#eval calculatePostage (35/100) (25/100) (53/10)

end NUMINAMATH_CALUDE_postage_calculation_l911_91110


namespace NUMINAMATH_CALUDE_side_ratio_l911_91105

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def special_triangle (t : Triangle) : Prop :=
  t.A > t.B ∧ t.B > t.C ∧  -- A is largest, C is smallest
  t.A = 2 * t.C ∧          -- A = 2C
  t.a + t.c = 2 * t.b      -- a + c = 2b

-- Theorem statement
theorem side_ratio (t : Triangle) (h : special_triangle t) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 6*k ∧ t.b = 5*k ∧ t.c = 3*k :=
sorry

end NUMINAMATH_CALUDE_side_ratio_l911_91105


namespace NUMINAMATH_CALUDE_cyclic_sum_equals_two_l911_91188

theorem cyclic_sum_equals_two (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 + a + a*b) / (1 + a + a*b + a*b*c) + 
  (1 + b + b*c) / (1 + b + b*c + b*c*d) + 
  (1 + c + c*d) / (1 + c + c*d + c*d*a) + 
  (1 + d + d*a) / (1 + d + d*a + d*a*b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_equals_two_l911_91188


namespace NUMINAMATH_CALUDE_work_completion_l911_91187

theorem work_completion (days_first_group : ℝ) (men_second_group : ℕ) (days_second_group : ℝ) :
  days_first_group = 25 →
  men_second_group = 20 →
  days_second_group = 18.75 →
  ∃ (men_first_group : ℕ), 
    men_first_group * days_first_group = men_second_group * days_second_group ∧
    men_first_group = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_l911_91187


namespace NUMINAMATH_CALUDE_triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l911_91118

/-- A triangle with side lengths satisfying a² + b² + c² = ab + bc + ca is equilateral -/
theorem triangle_equality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : a^2 + b^2 + c^2 = a*b + b*c + c*a) : a = b ∧ b = c := by
  sorry

/-- An equilateral triangle is isosceles -/
theorem equilateral_is_isosceles (a b c : ℝ) (h : a = b ∧ b = c) : 
  a = b ∨ b = c ∨ a = c := by
  sorry

/-- An equilateral triangle is acute-angled -/
theorem equilateral_is_acute (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 := by
  sorry

/-- An equilateral triangle is oblique (not right-angled) -/
theorem equilateral_is_oblique (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l911_91118


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l911_91109

theorem absolute_value_inequality (b : ℝ) (h₁ : b > 0) :
  (∃ x : ℝ, |2*x - 8| + |2*x - 6| < b) → b > 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l911_91109


namespace NUMINAMATH_CALUDE_eight_hash_six_l911_91146

/-- Definition of the # operation -/
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

/-- First condition: r # 0 = r + 1 -/
axiom hash_zero (r : ℝ) : hash r 0 = r + 1

/-- Second condition: r # s = s # r -/
axiom hash_comm (r s : ℝ) : hash r s = hash s r

/-- Third condition: (r + 1) # s = (r # s) + s + 2 -/
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

/-- The main theorem to prove -/
theorem eight_hash_six : hash 8 6 = 69 :=
  sorry

end NUMINAMATH_CALUDE_eight_hash_six_l911_91146


namespace NUMINAMATH_CALUDE_odd_number_probability_l911_91131

-- Define a fair six-sided die
def FairDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set of odd numbers on the die
def OddNumbers : Finset ℕ := {1, 3, 5}

-- Theorem: The probability of rolling an odd number is 1/2
theorem odd_number_probability :
  (Finset.card OddNumbers : ℚ) / (Finset.card FairDie : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_probability_l911_91131


namespace NUMINAMATH_CALUDE_operation_probability_l911_91163

/-- An operation that randomly changes a positive integer to a smaller nonnegative integer -/
def operation (n : ℕ+) : ℕ := sorry

/-- The probability of choosing any specific smaller number during the operation -/
def transition_prob (n k : ℕ) : ℝ := sorry

/-- The probability of encountering specific numbers during the operation process -/
def encounter_prob (start : ℕ+) (targets : List ℕ) : ℝ := sorry

theorem operation_probability :
  encounter_prob 2019 [10, 100, 1000] = 1 / 2019000000 := by sorry

end NUMINAMATH_CALUDE_operation_probability_l911_91163


namespace NUMINAMATH_CALUDE_solution_difference_l911_91168

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 + 4*r - 21) = r + 3) →
  ((6 * s - 18) / (s^2 + 4*s - 21) = s + 3) →
  r ≠ s →
  r > s →
  r - s = 12 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l911_91168


namespace NUMINAMATH_CALUDE_largest_712_triple_l911_91164

/-- Converts a number from base 7 to base 10 --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 12 --/
def decimalToBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 7-12 triple --/
def is712Triple (n : ℕ) : Prop :=
  decimalToBase12 n = 3 * base7ToDecimal n

/-- The largest 7-12 triple --/
def largestTriple : ℕ := 450

theorem largest_712_triple :
  is712Triple largestTriple ∧
  ∀ n : ℕ, n > largestTriple → ¬is712Triple n := by sorry

end NUMINAMATH_CALUDE_largest_712_triple_l911_91164


namespace NUMINAMATH_CALUDE_salt_concentration_after_dilution_l911_91189

/-- Calculates the final salt concentration after adding water to a salt solution -/
theorem salt_concentration_after_dilution
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 56)
  (h2 : initial_concentration = 0.1)
  (h3 : water_added = 14) :
  let salt_amount := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := salt_amount / final_volume
  final_concentration = 0.08 := by sorry

end NUMINAMATH_CALUDE_salt_concentration_after_dilution_l911_91189


namespace NUMINAMATH_CALUDE_solution_x_volume_l911_91124

/-- Proves that the volume of solution x is 50 milliliters, given the conditions of the mixing problem. -/
theorem solution_x_volume (x_concentration : Real) (y_concentration : Real) (y_volume : Real) (final_concentration : Real) :
  x_concentration = 0.10 →
  y_concentration = 0.30 →
  y_volume = 150 →
  final_concentration = 0.25 →
  ∃ (x_volume : Real),
    x_volume = 50 ∧
    (x_concentration * x_volume + y_concentration * y_volume) / (x_volume + y_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_solution_x_volume_l911_91124


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l911_91113

theorem max_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum1 : x/y + y/z + z/x = 3) (hsum2 : x + y + z = 6) :
  ∃ (M : ℝ), ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    a/b + b/c + c/a = 3 → a + b + c = 6 →
    y/x + z/y + x/z ≤ M ∧ M = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l911_91113


namespace NUMINAMATH_CALUDE_unique_solution_natural_numbers_l911_91195

theorem unique_solution_natural_numbers : 
  ∃! (a b : ℕ), a^b + a + b = b^a ∧ a = 5 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_natural_numbers_l911_91195


namespace NUMINAMATH_CALUDE_characterization_of_n_l911_91148

/-- A positive integer is square-free if it is not divisible by any perfect square greater than 1 -/
def IsSquareFree (n : ℕ+) : Prop :=
  ∀ (d : ℕ+), d * d ∣ n → d = 1

/-- The condition that for all positive integers x and y, if n divides x^n - y^n, then n^2 divides x^n - y^n -/
def Condition (n : ℕ+) : Prop :=
  ∀ (x y : ℕ+), n ∣ (x ^ n.val - y ^ n.val) → n.val * n.val ∣ (x ^ n.val - y ^ n.val)

/-- The main theorem stating the characterization of n satisfying the condition -/
theorem characterization_of_n (n : ℕ+) :
  Condition n ↔ (∃ (m : ℕ+), IsSquareFree m ∧ (n = m ∨ n = 2 * m)) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_n_l911_91148


namespace NUMINAMATH_CALUDE_triangle_side_value_l911_91112

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a * b = 60 →
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  R = Real.sqrt 3 →
  c = 2 * R * Real.sin C →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l911_91112


namespace NUMINAMATH_CALUDE_rescue_center_dogs_l911_91174

/-- Calculates the number of remaining dogs after a series of additions and adoptions. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given the specific numbers from the problem, 
    the number of remaining dogs is 200. -/
theorem rescue_center_dogs : 
  remaining_dogs 200 100 40 60 = 200 := by
  sorry

#eval remaining_dogs 200 100 40 60

end NUMINAMATH_CALUDE_rescue_center_dogs_l911_91174


namespace NUMINAMATH_CALUDE_tesseract_triangles_l911_91176

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_triangles_l911_91176


namespace NUMINAMATH_CALUDE_candy_ratio_l911_91126

theorem candy_ratio (m_and_m : ℕ) (starburst : ℕ) : 
  (7 : ℕ) * starburst = (4 : ℕ) * m_and_m → m_and_m = 56 → starburst = 32 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_l911_91126


namespace NUMINAMATH_CALUDE_function_equation_l911_91177

theorem function_equation (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = 1 - x^2) →
  (∀ x : ℝ, x^2 * f x + f (1 - x) = 2*x - x^4) := by
sorry

end NUMINAMATH_CALUDE_function_equation_l911_91177


namespace NUMINAMATH_CALUDE_equation_solution_l911_91154

theorem equation_solution : 
  ∃ x : ℚ, (x - 50) / 3 = (5 - 3 * x) / 4 + 2 ∧ x = 287 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l911_91154


namespace NUMINAMATH_CALUDE_central_symmetry_intersection_condition_l911_91165

/-- Two functions are centrally symmetric and intersect at one point -/
def centrally_symmetric_one_intersection (a b c d : ℝ) : Prop :=
  let f := fun x => 2 * a + 1 / (x - b)
  let g := fun x => 2 * c + 1 / (x - d)
  let center := ((b + d) / 2, a + c)
  ∃! x, f x = g x ∧ 
    ∀ y, f ((b + d) - y) = g y ∧ 
         g ((b + d) - y) = f y

/-- The main theorem -/
theorem central_symmetry_intersection_condition (a b c d : ℝ) :
  centrally_symmetric_one_intersection a b c d ↔ (a - c) * (b - d) = 2 :=
by sorry


end NUMINAMATH_CALUDE_central_symmetry_intersection_condition_l911_91165


namespace NUMINAMATH_CALUDE_wrapping_paper_area_for_specific_box_l911_91198

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the square wrapping paper needed for a given box. -/
def wrappingPaperArea (box : BoxDimensions) : ℝ :=
  4 * box.width ^ 2

/-- Theorem stating that for a box with dimensions a × 2a × a, 
    the area of the square wrapping paper is 4a². -/
theorem wrapping_paper_area_for_specific_box (a : ℝ) (h : a > 0) :
  let box : BoxDimensions := ⟨a, 2*a, a⟩
  wrappingPaperArea box = 4 * a ^ 2 := by
  sorry

#check wrapping_paper_area_for_specific_box

end NUMINAMATH_CALUDE_wrapping_paper_area_for_specific_box_l911_91198


namespace NUMINAMATH_CALUDE_remainder_of_sum_l911_91127

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_l911_91127


namespace NUMINAMATH_CALUDE_parabola_roots_difference_l911_91135

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (h k : ℝ), h = 3 ∧ k = -3 ∧ ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  parabola a b c 5 = 9 →
  (∃ (m n : ℝ), m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  ∃ (m n : ℝ), m - n = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_roots_difference_l911_91135


namespace NUMINAMATH_CALUDE_gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l911_91143

/-- The cost of a gold heart necklace given the following conditions:
  * Bracelets cost $15 each
  * Personalized coffee mug costs $20
  * Raine buys 3 bracelets, 2 gold heart necklaces, and 1 coffee mug
  * Raine pays with a $100 bill and gets $15 change
-/
theorem gold_heart_necklace_cost : ℝ :=
  let bracelet_cost : ℝ := 15
  let mug_cost : ℝ := 20
  let num_bracelets : ℕ := 3
  let num_necklaces : ℕ := 2
  let num_mugs : ℕ := 1
  let payment : ℝ := 100
  let change : ℝ := 15
  let total_spent : ℝ := payment - change
  let necklace_cost : ℝ := (total_spent - (bracelet_cost * num_bracelets + mug_cost * num_mugs)) / num_necklaces
  10

theorem gold_heart_necklace_cost_proof : gold_heart_necklace_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l911_91143


namespace NUMINAMATH_CALUDE_three_number_product_l911_91170

theorem three_number_product (a b c m : ℚ) : 
  a + b + c = 210 →
  8 * a = m →
  b - 12 = m →
  c + 12 = m →
  a * b * c = 58116480 / 4913 := by
sorry

end NUMINAMATH_CALUDE_three_number_product_l911_91170


namespace NUMINAMATH_CALUDE_unique_n_for_integer_Sn_l911_91149

theorem unique_n_for_integer_Sn : ∃! (n : ℕ+), ∃ (m : ℕ), 
  n.val > 0 ∧ m^2 = 17^2 + n.val^2 ∧ 
  ∀ (k : ℕ+), k ≠ n → ¬∃ (l : ℕ), l^2 = 17^2 + k.val^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_integer_Sn_l911_91149


namespace NUMINAMATH_CALUDE_complement_determines_set_l911_91151

def U : Set Nat := {0, 1, 2, 4}

theorem complement_determines_set 
  (h : Set.compl {1, 2} = {0, 4}) : 
  ∃ A : Set Nat, A ⊆ U ∧ Set.compl A = {1, 2} ∧ A = {0, 4} := by
  sorry

#check complement_determines_set

end NUMINAMATH_CALUDE_complement_determines_set_l911_91151


namespace NUMINAMATH_CALUDE_f_f_eq_f_solutions_l911_91132

def f (x : ℝ) := x^2 - 2*x

theorem f_f_eq_f_solutions :
  {x : ℝ | f (f x) = f x} = {0, 2, -1, 3} := by sorry

end NUMINAMATH_CALUDE_f_f_eq_f_solutions_l911_91132


namespace NUMINAMATH_CALUDE_product_of_sums_l911_91199

theorem product_of_sums (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) :
  (x + 2) * (y + 2) = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_sums_l911_91199


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l911_91192

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perpendicular n α) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l911_91192


namespace NUMINAMATH_CALUDE_cosine_angle_vectors_l911_91193

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : 2 * ‖a‖ = 3 * ‖b‖) (h2 : ‖a - 2•b‖ = ‖a + b‖) :
  inner a b / (‖a‖ * ‖b‖) = 1/3 := by sorry

end NUMINAMATH_CALUDE_cosine_angle_vectors_l911_91193


namespace NUMINAMATH_CALUDE_negative_real_inequality_l911_91134

theorem negative_real_inequality (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
  x * y * z / ((1 + 5*x) * (4*x + 3*y) * (5*y + 6*z) * (z + 18)) ≤ 1 / 5120 := by
  sorry

end NUMINAMATH_CALUDE_negative_real_inequality_l911_91134


namespace NUMINAMATH_CALUDE_one_third_of_6_3_l911_91100

theorem one_third_of_6_3 : (6.3 : ℚ) / 3 = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_6_3_l911_91100


namespace NUMINAMATH_CALUDE_max_discussions_left_l911_91138

/-- Represents a group of politicians at a summit --/
structure PoliticianGroup where
  size : Nat
  has_talked : Fin size → Fin size → Bool
  all_pairs_plan_to_talk : ∀ i j, i ≠ j → has_talked i j = false → True
  four_politician_condition : ∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (has_talked a b ∧ has_talked a c ∧ has_talked a d) ∨
    (has_talked b a ∧ has_talked b c ∧ has_talked b d) ∨
    (has_talked c a ∧ has_talked c b ∧ has_talked c d) ∨
    (has_talked d a ∧ has_talked d b ∧ has_talked d c)

/-- The theorem stating the maximum number of discussions yet to be held --/
theorem max_discussions_left (g : PoliticianGroup) (h : g.size = 2018) :
  (∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ¬g.has_talked a b ∧ ¬g.has_talked b c ∧ ¬g.has_talked a c) ∧
  (∀ a b c d, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    g.has_talked a b ∨ g.has_talked b c ∨ g.has_talked a c ∨
    g.has_talked a d ∨ g.has_talked b d ∨ g.has_talked c d) :=
by sorry

end NUMINAMATH_CALUDE_max_discussions_left_l911_91138


namespace NUMINAMATH_CALUDE_buyers_both_mixes_l911_91171

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers purchasing cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers purchasing muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the probability of selecting a buyer who purchases neither cake mix nor muffin mix
def prob_neither : ℚ := 27 / 100

-- Theorem to prove
theorem buyers_both_mixes (both_mixes : ℕ) : 
  (cake_mix_buyers + muffin_mix_buyers - both_mixes = total_buyers * (1 - prob_neither)) →
  both_mixes = 17 := by
  sorry

end NUMINAMATH_CALUDE_buyers_both_mixes_l911_91171


namespace NUMINAMATH_CALUDE_min_mines_is_23_l911_91128

/-- Represents the state of a square on the Minesweeper board -/
inductive SquareState
| Unopened
| Opened (n : Nat)

/-- Represents a Minesweeper board -/
def Board := Matrix (Fin 11) (Fin 13) SquareState

/-- Checks if a given position is valid on the board -/
def isValidPos (row col : Nat) : Bool :=
  row < 11 && col < 13

/-- Returns the number of mines in the neighboring squares -/
def countNeighborMines (b : Board) (row col : Nat) : Nat :=
  sorry

/-- Checks if a board configuration is valid according to Minesweeper rules -/
def isValidBoard (b : Board) : Prop :=
  ∀ row col, isValidPos row col →
    match b row col with
    | SquareState.Opened n => countNeighborMines b row col = n
    | SquareState.Unopened => True

/-- Creates the specific Minesweeper board configuration from the problem -/
def createProblemBoard : Board :=
  sorry

/-- Counts the total number of mines on the board -/
def countMines (b : Board) : Nat :=
  sorry

/-- Main theorem: The minimum number of mines on the given Minesweeper board is 23 -/
theorem min_mines_is_23 (b : Board) :
  isValidBoard b → createProblemBoard = b → countMines b ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_min_mines_is_23_l911_91128


namespace NUMINAMATH_CALUDE_cricketer_average_score_l911_91123

/-- 
Given a cricketer whose average score increases by 4 after scoring 95 runs in the 19th inning,
this theorem proves that the cricketer's average score after 19 innings is 23 runs per inning.
-/
theorem cricketer_average_score 
  (initial_average : ℝ) 
  (score_increase : ℝ) 
  (runs_19th_inning : ℕ) :
  score_increase = 4 →
  runs_19th_inning = 95 →
  (18 * initial_average + runs_19th_inning) / 19 = initial_average + score_increase →
  initial_average + score_increase = 23 :=
by
  sorry

#check cricketer_average_score

end NUMINAMATH_CALUDE_cricketer_average_score_l911_91123


namespace NUMINAMATH_CALUDE_value_of_a_l911_91115

theorem value_of_a (a b c : ℤ) 
  (sum_ab : a + b = 2)
  (opposite_bc : b + c = 0)
  (abs_c : |c| = 1) :
  a = 3 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_value_of_a_l911_91115


namespace NUMINAMATH_CALUDE_steel_copper_weight_difference_l911_91122

/-- Represents the weight of a metal bar in kilograms. -/
structure MetalBar where
  weight : ℝ

/-- The container with metal bars. -/
structure Container where
  steel : MetalBar
  tin : MetalBar
  copper : MetalBar
  count : ℕ
  totalWeight : ℝ

/-- Theorem stating the weight difference between steel and copper bars. -/
theorem steel_copper_weight_difference (c : Container) : 
  c.steel.weight - c.copper.weight = 20 :=
  by
  have h1 : c.steel.weight = 2 * c.tin.weight := sorry
  have h2 : c.copper.weight = 90 := sorry
  have h3 : c.count = 20 := sorry
  have h4 : c.totalWeight = 5100 := sorry
  have h5 : c.count * (c.steel.weight + c.tin.weight + c.copper.weight) = c.totalWeight := sorry
  sorry

#check steel_copper_weight_difference

end NUMINAMATH_CALUDE_steel_copper_weight_difference_l911_91122


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l911_91159

theorem point_on_terminal_side (m : ℝ) (α : ℝ) :
  (2 : ℝ) / Real.sqrt (m^2 + 4) = (1 : ℝ) / 3 →
  m = 4 * Real.sqrt 2 ∨ m = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l911_91159


namespace NUMINAMATH_CALUDE_base2_110110110_to_base8_l911_91178

/-- Converts a base 2 number to base 8 -/
def base2ToBase8 (n : ℕ) : ℕ :=
  sorry

theorem base2_110110110_to_base8 :
  base2ToBase8 0b110110110 = 0o666 := by
  sorry

end NUMINAMATH_CALUDE_base2_110110110_to_base8_l911_91178


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_quadratic_inequality_not_sufficient_l911_91142

theorem quadratic_inequality_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → a ≥ 0 :=
by sorry

theorem quadratic_inequality_not_sufficient (a : ℝ) :
  a ≥ 0 → ¬(∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_condition_quadratic_inequality_not_sufficient_l911_91142


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l911_91147

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem complement_of_A_in_U :
  (U \ A) = {x | (1 ≤ x ∧ x < 2) ∨ x = 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l911_91147


namespace NUMINAMATH_CALUDE_min_value_expression_l911_91106

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 ∧
  ∃ x y, x > 1 ∧ y > 1 ∧ (x^3 / (y - 1)) + (y^3 / (x - 1)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l911_91106


namespace NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l911_91101

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) : 
  total_sunglasses = 60 → 
  total_caps = 40 → 
  prob_cap_given_sunglasses = 1/3 → 
  (total_sunglasses * prob_cap_given_sunglasses : ℚ) / total_caps = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_and_caps_probability_l911_91101


namespace NUMINAMATH_CALUDE_largest_integer_squared_less_than_ten_million_l911_91102

theorem largest_integer_squared_less_than_ten_million :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 10000000 ∧ ∀ (m : ℕ), m > n → m^2 ≥ 10000000 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_squared_less_than_ten_million_l911_91102


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l911_91150

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (|m| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l911_91150


namespace NUMINAMATH_CALUDE_exists_polyhedron_with_properties_l911_91155

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  volume : ℝ
  surfaceArea : ℝ
  volumeBelowWater : ℝ
  surfaceAreaAboveWater : ℝ

/-- Theorem stating the existence of a convex polyhedron with the given properties -/
theorem exists_polyhedron_with_properties :
  ∃ (p : ConvexPolyhedron),
    p.volumeBelowWater = 0.9 * p.volume ∧
    p.surfaceAreaAboveWater > 0.5 * p.surfaceArea :=
sorry

end NUMINAMATH_CALUDE_exists_polyhedron_with_properties_l911_91155


namespace NUMINAMATH_CALUDE_calculator_cost_proof_l911_91182

theorem calculator_cost_proof (basic scientific graphing : ℝ) 
  (h1 : scientific = 2 * basic)
  (h2 : graphing = 3 * scientific)
  (h3 : basic + scientific + graphing = 72) :
  basic = 8 := by
sorry

end NUMINAMATH_CALUDE_calculator_cost_proof_l911_91182


namespace NUMINAMATH_CALUDE_factor_expression_l911_91136

theorem factor_expression (y : ℝ) : 3 * y * (y - 5) + 4 * (y - 5) = (3 * y + 4) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l911_91136
