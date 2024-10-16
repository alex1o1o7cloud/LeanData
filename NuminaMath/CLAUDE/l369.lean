import Mathlib

namespace NUMINAMATH_CALUDE_reflected_ray_passes_through_C_l369_36936

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the reflected ray equation
def reflected_ray_equation (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Theorem statement
theorem reflected_ray_passes_through_C : 
  ∃ C : ℝ × ℝ, C.1 = 1 ∧ C.2 = 4 ∧ reflected_ray_equation C.1 C.2 := by sorry

end NUMINAMATH_CALUDE_reflected_ray_passes_through_C_l369_36936


namespace NUMINAMATH_CALUDE_power_of_power_l369_36914

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l369_36914


namespace NUMINAMATH_CALUDE_matrix_inverse_problem_l369_36911

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_inverse_problem (B : Matrix n n ℝ) (h_inv : Invertible B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) :
  B + 10 • B⁻¹ = (160 / 15 : ℝ) • 1 := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_problem_l369_36911


namespace NUMINAMATH_CALUDE_meet_once_l369_36921

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def numberOfMeetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michaelSpeed = 4)
  (h2 : scenario.truckSpeed = 8)
  (h3 : scenario.pailDistance = 300)
  (h4 : scenario.truckStopTime = 45)
  (h5 : scenario.initialDistance = 300) : 
  numberOfMeetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l369_36921


namespace NUMINAMATH_CALUDE_vector_b_value_l369_36945

theorem vector_b_value (a b : ℝ × ℝ × ℝ) :
  a = (4, 0, -2) →
  a - b = (0, 1, -2) →
  b = (4, -1, 0) := by
sorry

end NUMINAMATH_CALUDE_vector_b_value_l369_36945


namespace NUMINAMATH_CALUDE_value_of_y_l369_36959

theorem value_of_y : ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l369_36959


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_5_6_2_l369_36980

theorem largest_four_digit_divisible_by_5_6_2 :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 9990 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_5_6_2_l369_36980


namespace NUMINAMATH_CALUDE_sequence_problem_l369_36912

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 5 * b 9 = 16 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l369_36912


namespace NUMINAMATH_CALUDE_min_value_theorem_l369_36918

theorem min_value_theorem (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l369_36918


namespace NUMINAMATH_CALUDE_prob_10_or_9_prob_at_least_7_l369_36987

/-- Represents the probabilities of hitting different rings in a shooting event -/
structure ShootingProbabilities where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ
  ring7 : ℝ
  below7 : ℝ

/-- The probabilities sum to 1 -/
axiom prob_sum_to_one (p : ShootingProbabilities) : 
  p.ring10 + p.ring9 + p.ring8 + p.ring7 + p.below7 = 1

/-- All probabilities are non-negative -/
axiom prob_non_negative (p : ShootingProbabilities) : 
  p.ring10 ≥ 0 ∧ p.ring9 ≥ 0 ∧ p.ring8 ≥ 0 ∧ p.ring7 ≥ 0 ∧ p.below7 ≥ 0

/-- Given probabilities for a shooting event -/
def shooter_probs : ShootingProbabilities := {
  ring10 := 0.1,
  ring9 := 0.2,
  ring8 := 0.3,
  ring7 := 0.3,
  below7 := 0.1
}

/-- Theorem: The probability of hitting the 10 or 9 ring is 0.3 -/
theorem prob_10_or_9 : shooter_probs.ring10 + shooter_probs.ring9 = 0.3 := by sorry

/-- Theorem: The probability of hitting at least the 7 ring is 0.9 -/
theorem prob_at_least_7 : 1 - shooter_probs.below7 = 0.9 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_9_prob_at_least_7_l369_36987


namespace NUMINAMATH_CALUDE_binomial_sum_equals_sixteen_l369_36986

theorem binomial_sum_equals_sixteen (h : (Complex.exp (Complex.I * Real.pi / 4))^10 = Complex.I) :
  Nat.choose 10 1 - Nat.choose 10 3 + (Nat.choose 10 5 / 2) = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_sixteen_l369_36986


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l369_36983

theorem difference_of_squares_division : (324^2 - 300^2) / 24 = 624 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l369_36983


namespace NUMINAMATH_CALUDE_min_value_theorem_l369_36975

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l369_36975


namespace NUMINAMATH_CALUDE_reflect_point_over_x_axis_l369_36943

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the x-axis -/
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflect_point_over_x_axis :
  let P : Point := { x := -6, y := -9 }
  reflectOverXAxis P = { x := -6, y := 9 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_over_x_axis_l369_36943


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l369_36900

theorem gcd_of_squares_sum : Nat.gcd (12^2 + 23^2 + 34^2) (13^2 + 22^2 + 35^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l369_36900


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l369_36908

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = -3 ∧ m = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l369_36908


namespace NUMINAMATH_CALUDE_thousand_gon_triangles_l369_36952

/-- Given a polygon with n sides and m internal points, calculates the number of triangles formed when the points are connected to each other and to the vertices of the polygon. -/
def triangles_in_polygon (n : ℕ) (m : ℕ) : ℕ :=
  n + 2 * m - 2

/-- Theorem stating that in a 1000-sided polygon with 500 internal points, 1998 triangles are formed. -/
theorem thousand_gon_triangles :
  triangles_in_polygon 1000 500 = 1998 := by
  sorry

end NUMINAMATH_CALUDE_thousand_gon_triangles_l369_36952


namespace NUMINAMATH_CALUDE_williams_land_ratio_l369_36940

/-- The ratio of an individual's tax payment to the total tax collected equals the ratio of their taxable land to the total taxable land -/
axiom tax_ratio_equals_land_ratio {total_tax individual_tax total_land individual_land : ℚ} :
  individual_tax / total_tax = individual_land / total_land

/-- Given the total farm tax and an individual's farm tax, prove that the ratio of the individual's
    taxable land to the total taxable land is 1/8 -/
theorem williams_land_ratio (total_tax individual_tax : ℚ)
    (h1 : total_tax = 3840)
    (h2 : individual_tax = 480) :
    ∃ (total_land individual_land : ℚ),
      individual_land / total_land = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_williams_land_ratio_l369_36940


namespace NUMINAMATH_CALUDE_equation_represents_circle_l369_36901

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 0)^2 + (y - 0)^2 = 25

-- Define what a circle is in terms of its equation
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    ∀ (x y : ℝ), f x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

-- Theorem statement
theorem equation_represents_circle : is_circle equation := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_circle_l369_36901


namespace NUMINAMATH_CALUDE_kyle_origami_stars_l369_36995

theorem kyle_origami_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) :
  initial_bottles = 4 →
  additional_bottles = 5 →
  stars_per_bottle = 25 →
  (initial_bottles + additional_bottles) * stars_per_bottle = 225 := by
  sorry

end NUMINAMATH_CALUDE_kyle_origami_stars_l369_36995


namespace NUMINAMATH_CALUDE_flower_pot_price_difference_l369_36917

theorem flower_pot_price_difference 
  (n : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (h1 : n = 6) 
  (h2 : total_cost = 39/5) 
  (h3 : largest_pot_cost = 77/40) : 
  ∃ (d : ℚ), d = 1/4 ∧ 
  ∃ (x : ℚ), 
    (x + (n - 1) * d = largest_pot_cost) ∧ 
    (n * x + (n * (n - 1) / 2) * d = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_flower_pot_price_difference_l369_36917


namespace NUMINAMATH_CALUDE_perfect_square_arrangement_l369_36961

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that represents a permutation of numbers from 1 to n -/
def permutation (n : ℕ) := Fin n → Fin n

/-- A property that checks if a permutation satisfies the perfect square sum condition -/
def valid_permutation (n : ℕ) (p : permutation n) : Prop :=
  ∀ i : Fin n, is_perfect_square (i.val + 1 + (p i).val + 1)

theorem perfect_square_arrangement :
  (∃ p : permutation 9, valid_permutation 9 p) ∧
  (¬ ∃ p : permutation 11, valid_permutation 11 p) ∧
  (∃ p : permutation 1996, valid_permutation 1996 p) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_arrangement_l369_36961


namespace NUMINAMATH_CALUDE_range_of_f_l369_36935

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l369_36935


namespace NUMINAMATH_CALUDE_double_seven_eighths_of_48_l369_36989

theorem double_seven_eighths_of_48 : 2 * (7 / 8 * 48) = 84 := by
  sorry

end NUMINAMATH_CALUDE_double_seven_eighths_of_48_l369_36989


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l369_36930

theorem consecutive_odd_integers_sum (x : ℤ) :
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ x + z = 150) →
  x + (x + 2) + (x + 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l369_36930


namespace NUMINAMATH_CALUDE_jose_alisson_difference_l369_36997

/-- Represents the scores of three students in a test -/
structure TestScores where
  jose : ℕ
  meghan : ℕ
  alisson : ℕ

/-- Properties of the test and scores -/
def valid_scores (s : TestScores) : Prop :=
  s.meghan = s.jose - 20 ∧
  s.jose > s.alisson ∧
  s.jose = 90 ∧
  s.jose + s.meghan + s.alisson = 210

/-- Theorem stating the difference between Jose's and Alisson's scores -/
theorem jose_alisson_difference (s : TestScores) 
  (h : valid_scores s) : s.jose - s.alisson = 40 := by
  sorry

end NUMINAMATH_CALUDE_jose_alisson_difference_l369_36997


namespace NUMINAMATH_CALUDE_balloon_difference_l369_36955

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l369_36955


namespace NUMINAMATH_CALUDE_max_container_volume_height_for_max_volume_l369_36933

/-- Represents the volume of a rectangular container as a function of one side length --/
def containerVolume (x : ℝ) : ℝ := x * (x + 0.5) * (3.45 - x)

/-- The total length of the steel strip used for the container frame --/
def totalLength : ℝ := 14.8

/-- Theorem stating the maximum volume of the container --/
theorem max_container_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  ∀ (y : ℝ), y > 0 → y < 3.45 → containerVolume y ≤ containerVolume x :=
sorry

/-- Theorem stating the height that achieves the maximum volume --/
theorem height_for_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3.45 ∧
  containerVolume x = 3.675 ∧
  (3.45 - x) = 2.45 :=
sorry

end NUMINAMATH_CALUDE_max_container_volume_height_for_max_volume_l369_36933


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percent_l369_36939

theorem shopkeeper_profit_percent
  (initial_value : ℝ)
  (theft_percent : ℝ)
  (overall_loss_percent : ℝ)
  (h_theft : theft_percent = 20)
  (h_loss : overall_loss_percent = 12)
  (h_initial_positive : initial_value > 0) :
  let remaining_value := initial_value * (1 - theft_percent / 100)
  let sale_value := initial_value * (1 - overall_loss_percent / 100)
  let profit := sale_value - remaining_value
  profit / remaining_value * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percent_l369_36939


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l369_36929

/-- Two vectors in R² are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2) ∨ w = (k * v.1, k * v.2)

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ (x : ℝ), collinear (x, 1) (4, x) → x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l369_36929


namespace NUMINAMATH_CALUDE_combined_distance_theorem_l369_36947

/-- Represents the four lakes in the migration sequence -/
inductive Lake : Type
| Jim : Lake
| Disney : Lake
| London : Lake
| Everest : Lake

/-- The number of birds in the group -/
def num_birds : ℕ := 25

/-- The number of migration sequences completed in a year -/
def sequences_per_year : ℕ := 2

/-- The distance between two lakes in miles -/
def distance (a b : Lake) : ℕ :=
  match a, b with
  | Lake.Jim, Lake.Disney => 42
  | Lake.Disney, Lake.London => 57
  | Lake.London, Lake.Everest => 65
  | Lake.Everest, Lake.Jim => 70
  | _, _ => 0  -- For other combinations, return 0

/-- The total distance of one migration sequence -/
def sequence_distance : ℕ :=
  distance Lake.Jim Lake.Disney +
  distance Lake.Disney Lake.London +
  distance Lake.London Lake.Everest +
  distance Lake.Everest Lake.Jim

/-- Theorem: The combined distance traveled by all birds in a year is 11,700 miles -/
theorem combined_distance_theorem :
  num_birds * sequences_per_year * sequence_distance = 11700 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_theorem_l369_36947


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l369_36976

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -1 + Real.sqrt 2 ∧ 
  x₂ = -1 - Real.sqrt 2 ∧ 
  x₁^2 + 2*x₁ - 1 = 0 ∧ 
  x₂^2 + 2*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l369_36976


namespace NUMINAMATH_CALUDE_blueberry_pancakes_l369_36907

theorem blueberry_pancakes (total : ℕ) (banana : ℕ) (plain : ℕ)
  (h1 : total = 67)
  (h2 : banana = 24)
  (h3 : plain = 23) :
  total - banana - plain = 20 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_pancakes_l369_36907


namespace NUMINAMATH_CALUDE_cos_2x_min_value_in_interval_l369_36993

theorem cos_2x_min_value_in_interval :
  ∃ x ∈ Set.Ioo 0 π, ∀ y ∈ Set.Ioo 0 π, Real.cos (2 * x) ≤ Real.cos (2 * y) ∧
  Real.cos (2 * x) = -1 :=
sorry

end NUMINAMATH_CALUDE_cos_2x_min_value_in_interval_l369_36993


namespace NUMINAMATH_CALUDE_laborer_income_proof_l369_36992

/-- Represents the monthly income of a laborer -/
def monthly_income : ℝ := 75

/-- Represents the debt after 6 months -/
def debt : ℝ := 30

theorem laborer_income_proof :
  let initial_period := 6
  let initial_monthly_expenditure := 80
  let later_period := 4
  let later_monthly_expenditure := 60
  let savings := 30
  (initial_period * monthly_income < initial_period * initial_monthly_expenditure) ∧
  (later_period * monthly_income = later_period * later_monthly_expenditure + debt + savings) →
  monthly_income = 75 := by
sorry

end NUMINAMATH_CALUDE_laborer_income_proof_l369_36992


namespace NUMINAMATH_CALUDE_no_divisible_by_19_l369_36960

def a (n : ℕ) : ℤ := 9 * 10^n + 11

theorem no_divisible_by_19 : ∀ k : ℕ, k < 3050 → ¬(19 ∣ a k) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_by_19_l369_36960


namespace NUMINAMATH_CALUDE_m_less_than_five_l369_36903

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem m_less_than_five
  (h_increasing : Increasing f)
  (h_inequality : ∀ m : ℝ, f (2 * m + 1) > f (3 * m - 4)) :
  ∀ m : ℝ, m < 5 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_five_l369_36903


namespace NUMINAMATH_CALUDE_race_finish_orders_l369_36978

-- Define the number of racers
def num_racers : ℕ := 4

-- Define the function to calculate the number of permutations
def num_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_finish_orders :
  num_permutations num_racers = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l369_36978


namespace NUMINAMATH_CALUDE_system_solution_l369_36922

variables (a b x y : ℝ)

theorem system_solution (h1 : x / (a - 2*b) - y / (a + 2*b) = (6*a*b) / (a^2 - 4*b^2))
                        (h2 : (x + y) / (a + 2*b) + (x - y) / (a - 2*b) = (2*(a^2 - a*b + 2*b^2)) / (a^2 - 4*b^2))
                        (h3 : a ≠ 2*b)
                        (h4 : a ≠ -2*b)
                        (h5 : a^2 ≠ 4*b^2) :
  x = a + b ∧ y = a - b :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l369_36922


namespace NUMINAMATH_CALUDE_dress_difference_l369_36990

theorem dress_difference (total_dresses : ℕ) (ana_dresses : ℕ) 
  (h1 : total_dresses = 48) 
  (h2 : ana_dresses = 15) : 
  total_dresses - ana_dresses - ana_dresses = 18 := by
  sorry

end NUMINAMATH_CALUDE_dress_difference_l369_36990


namespace NUMINAMATH_CALUDE_inequality_solution_set_l369_36915

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l369_36915


namespace NUMINAMATH_CALUDE_five_integer_solutions_l369_36906

theorem five_integer_solutions (x : ℤ) : 
  (∃ (S : Finset ℤ), (∀ y ∈ S, 5*y^2 + 19*y + 16 ≤ 20) ∧ 
                     (∀ z : ℤ, 5*z^2 + 19*z + 16 ≤ 20 → z ∈ S) ∧
                     S.card = 5) := by
  sorry

end NUMINAMATH_CALUDE_five_integer_solutions_l369_36906


namespace NUMINAMATH_CALUDE_system_solution_l369_36962

theorem system_solution (x y z : ℝ) :
  x + y + z = 2 ∧ x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l369_36962


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l369_36999

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.1111111111111111,
    given that y = 2 when x = 1. -/
theorem inverse_variation_proof (x y : ℝ) (k : ℝ) 
    (h1 : ∀ x y, x = k / (y * y))  -- x varies inversely as square of y
    (h2 : 1 = k / (2 * 2))         -- y = 2 when x = 1
    : y = 6 ↔ x = 0.1111111111111111 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l369_36999


namespace NUMINAMATH_CALUDE_additive_inverse_of_zero_l369_36969

theorem additive_inverse_of_zero : 
  (∀ x : ℝ, x + 0 = x) → 
  (∀ x : ℝ, x + (-x) = 0) → 
  (0 : ℝ) + 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_of_zero_l369_36969


namespace NUMINAMATH_CALUDE_quadratic_properties_l369_36973

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  (a - b + c = 0 → discriminant a b c ≥ 0) ∧
  (quadratic_equation a b c 1 ∧ quadratic_equation a b c 2 → 2*a - c = 0) ∧
  ((∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ z : ℝ, quadratic_equation a b c z) ∧
  (b = 2*a + c → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a b c x ∧ quadratic_equation a b c y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l369_36973


namespace NUMINAMATH_CALUDE_morning_routine_duration_l369_36977

def coffee_bagel_time : ℕ := 15

def paper_eating_time : ℕ := 2 * coffee_bagel_time

def total_routine_time : ℕ := coffee_bagel_time + paper_eating_time

theorem morning_routine_duration :
  total_routine_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_morning_routine_duration_l369_36977


namespace NUMINAMATH_CALUDE_fraction_subtraction_complex_fraction_division_l369_36934

-- Define a and b as real numbers
variable (a b : ℝ)

-- Assumption that a ≠ b
variable (h : a ≠ b)

-- First theorem
theorem fraction_subtraction : (b / (a - b)) - (a / (a - b)) = -1 := by sorry

-- Second theorem
theorem complex_fraction_division : 
  ((a^2 - a*b) / a^2) / ((a / b) - (b / a)) = b / (a + b) := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_complex_fraction_division_l369_36934


namespace NUMINAMATH_CALUDE_odd_function_property_l369_36981

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 2)))
  (h3 : f (-1) = -1) :
  f 2017 + f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l369_36981


namespace NUMINAMATH_CALUDE_existence_of_odd_digit_multiple_of_power_of_five_l369_36941

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def all_digits_odd (x : ℕ) : Prop :=
  ∀ d, d ∈ x.digits 10 → is_odd_digit d

theorem existence_of_odd_digit_multiple_of_power_of_five (n : ℕ) :
  n > 0 →
  ∃ x : ℕ,
    (x.digits 10).length = n ∧
    all_digits_odd x ∧
    x % (5^n) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_odd_digit_multiple_of_power_of_five_l369_36941


namespace NUMINAMATH_CALUDE_simplify_product_of_radicals_l369_36948

theorem simplify_product_of_radicals (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (27 * x) * Real.sqrt (32 * x) = 144 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_radicals_l369_36948


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l369_36931

theorem triangle_cosine_problem (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given condition
  ((Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C) →
  -- Conclusion
  Real.cos A = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l369_36931


namespace NUMINAMATH_CALUDE_ramanujan_number_l369_36968

/-- Given Hardy's complex number and the product of Hardy's and Ramanujan's numbers,
    prove that Ramanujan's number is 144/25 + 8/25i. -/
theorem ramanujan_number (h r : ℂ) : 
  h = 3 + 4*I ∧ r * h = 16 + 24*I → r = 144/25 + 8/25*I := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l369_36968


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l369_36925

/-- Proves that a cyclist traveling 136.4 km in 6 hours and 30 minutes has an average speed of approximately 5.83 m/s -/
theorem cyclist_average_speed :
  let distance_km : ℝ := 136.4
  let time_hours : ℝ := 6.5
  let distance_m : ℝ := distance_km * 1000
  let time_s : ℝ := time_hours * 3600
  let average_speed : ℝ := distance_m / time_s
  ∃ ε > 0, |average_speed - 5.83| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l369_36925


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l369_36902

theorem smallest_divisor_with_remainder (d : ℕ) : d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

theorem fifteen_satisfies_condition : 2021 % 15 = 11 := by
  sorry

theorem fifteen_is_smallest : ∀ d : ℕ, d > 0 ∧ 2021 % d = 11 → d ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_fifteen_satisfies_condition_fifteen_is_smallest_l369_36902


namespace NUMINAMATH_CALUDE_odometer_problem_l369_36923

theorem odometer_problem (a b c : ℕ) (ha : a ≥ 1) (hsum : a + b + c ≤ 10) 
  (hdiv : ∃ t : ℕ, (100 * a + 10 * c) - (100 * a + 10 * b + c) = 60 * t) :
  a^2 + b^2 + c^2 = 26 := by
sorry

end NUMINAMATH_CALUDE_odometer_problem_l369_36923


namespace NUMINAMATH_CALUDE_seeds_in_bag_l369_36916

-- Define the problem parameters
def seeds_per_ear : ℕ := 4
def price_per_ear : ℚ := 1 / 10
def cost_per_bag : ℚ := 1 / 2
def profit : ℚ := 40
def ears_sold : ℕ := 500

-- Define the theorem
theorem seeds_in_bag : 
  ∃ (seeds_per_bag : ℕ), 
    (ears_sold : ℚ) * price_per_ear - profit = 
    (ears_sold * seeds_per_ear : ℚ) / seeds_per_bag * cost_per_bag ∧ 
    seeds_per_bag = 100 :=
sorry

end NUMINAMATH_CALUDE_seeds_in_bag_l369_36916


namespace NUMINAMATH_CALUDE_cone_volume_not_equal_base_height_product_l369_36928

/-- The volume of a cone is not equal to the product of its base area and height. -/
theorem cone_volume_not_equal_base_height_product (S h : ℝ) (S_pos : S > 0) (h_pos : h > 0) :
  ∃ V : ℝ, V = (1/3) * S * h ∧ V ≠ S * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_not_equal_base_height_product_l369_36928


namespace NUMINAMATH_CALUDE_coprime_divisibility_theorem_l369_36988

theorem coprime_divisibility_theorem (a b : ℕ+) :
  (Nat.gcd (2 * a.val - 1) (2 * b.val + 1) = 1) →
  (a.val + b.val ∣ 4 * a.val * b.val + 1) →
  ∃ n : ℕ+, a.val = n.val ∧ b.val = n.val + 1 :=
by sorry

end NUMINAMATH_CALUDE_coprime_divisibility_theorem_l369_36988


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_6_l369_36963

-- Define the function f(x) = x³ - 2x + 2
def f (x : ℝ) : ℝ := x^3 - 2*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_2_6 :
  f 2 = 6 ∧ f' 2 = 10 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_6_l369_36963


namespace NUMINAMATH_CALUDE_quadratic_minimum_l369_36905

/-- The function f(x) = 5x^2 - 20x + 7 has a minimum value when x = 2 -/
theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), 5 * x^2 - 20 * x + 7 ≥ 5 * min_x^2 - 20 * min_x + 7 ∧ min_x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l369_36905


namespace NUMINAMATH_CALUDE_multiply_72515_9999_l369_36909

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72515_9999_l369_36909


namespace NUMINAMATH_CALUDE_triangle_side_product_l369_36920

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a+b)^2 - c^2 = 4 and C = 60°, then ab = 4/3 -/
theorem triangle_side_product (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (π/3) = 1/2) :
  a * b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_l369_36920


namespace NUMINAMATH_CALUDE_min_tangent_equals_radius_l369_36910

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def LineOfSymmetry (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- Point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- Tangent from a point to the circle -/
def Tangent (p : Point) (x y : ℝ) : ℝ :=
  sorry

/-- The radius of the circle -/
def Radius : ℝ :=
  2

theorem min_tangent_equals_radius (a b : ℝ) :
  ∀ (p : Point), p.a = a ∧ p.b = b →
  (∀ (x y : ℝ), Circle x y → LineOfSymmetry a b x y) →
  (∃ (x y : ℝ), Tangent p x y = Radius) ∧
  (∀ (x y : ℝ), Tangent p x y ≥ Radius) :=
sorry

end NUMINAMATH_CALUDE_min_tangent_equals_radius_l369_36910


namespace NUMINAMATH_CALUDE_fruit_market_problem_l369_36927

/-- Fruit market problem -/
theorem fruit_market_problem 
  (original_price : ℝ)
  (initial_profit : ℝ)
  (initial_sales : ℝ)
  (max_price_increase : ℝ)
  (sales_decrease_rate : ℝ)
  (reduced_price : ℝ)
  (target_daily_profit : ℝ)
  (h1 : original_price = 50)
  (h2 : initial_profit = 10)
  (h3 : initial_sales = 500)
  (h4 : max_price_increase = 8)
  (h5 : sales_decrease_rate = 20)
  (h6 : reduced_price = 32)
  (h7 : target_daily_profit = 6000) :
  ∃ (decrease_percentage : ℝ) (price_increase : ℝ) (max_profit : ℝ),
    decrease_percentage = 20 ∧
    price_increase = 5 ∧
    max_profit = 6125 ∧
    (1 - decrease_percentage / 100) ^ 2 * original_price = reduced_price ∧
    (initial_profit + price_increase) * (initial_sales - sales_decrease_rate * price_increase) = target_daily_profit ∧
    max_profit = (initial_profit + max_price_increase / 2) * (initial_sales - sales_decrease_rate * max_price_increase / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_market_problem_l369_36927


namespace NUMINAMATH_CALUDE_three_x_squared_y_squared_l369_36998

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 517) : 
  3*x^2*y^2 = 588 := by
sorry

end NUMINAMATH_CALUDE_three_x_squared_y_squared_l369_36998


namespace NUMINAMATH_CALUDE_check_payment_inequality_l369_36972

theorem check_payment_inequality (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 → 
  10 ≤ y ∧ y ≤ 99 → 
  100 * y + x - (100 * x + y) = 2156 →
  100 * y + x < 2 * (100 * x + y) := by
  sorry

end NUMINAMATH_CALUDE_check_payment_inequality_l369_36972


namespace NUMINAMATH_CALUDE_white_triangle_pairs_coincide_l369_36967

/-- Represents the number of triangles of each color in each half -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Theorem stating that given the conditions in the problem, 
    the number of coinciding white triangle pairs is 3 -/
theorem white_triangle_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 4)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 3) :
  ∃ (white_white_pairs : ℕ), white_white_pairs = 3 :=
sorry

end NUMINAMATH_CALUDE_white_triangle_pairs_coincide_l369_36967


namespace NUMINAMATH_CALUDE_triangle_inequality_l369_36954

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l369_36954


namespace NUMINAMATH_CALUDE_kubosdivision_l369_36951

theorem kubosdivision (k m : ℕ) (hk : k > 0) (hm : m > 0) (hkm : k > m) 
  (hdiv : (k^3 - m^3) ∣ (k * m * (k^2 - m^2))) : (k - m)^3 > 3 * k * m := by
  sorry

end NUMINAMATH_CALUDE_kubosdivision_l369_36951


namespace NUMINAMATH_CALUDE_train_travel_time_l369_36958

theorem train_travel_time (initial_time : ℝ) (increase1 increase2 increase3 : ℝ) :
  initial_time = 19.5 ∧ 
  increase1 = 0.3 ∧ 
  increase2 = 0.25 ∧ 
  increase3 = 0.2 → 
  initial_time / ((1 + increase1) * (1 + increase2) * (1 + increase3)) = 10 := by
sorry

end NUMINAMATH_CALUDE_train_travel_time_l369_36958


namespace NUMINAMATH_CALUDE_eight_sided_dice_divisible_by_four_probability_l369_36932

theorem eight_sided_dice_divisible_by_four_probability : 
  let dice_outcomes : Finset ℕ := Finset.range 8
  let divisible_by_four : Finset ℕ := {4, 8}
  let total_outcomes : ℕ := dice_outcomes.card * dice_outcomes.card
  let favorable_outcomes : ℕ := divisible_by_four.card * divisible_by_four.card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_eight_sided_dice_divisible_by_four_probability_l369_36932


namespace NUMINAMATH_CALUDE_rationalization_sum_l369_36956

theorem rationalization_sum (A B C D : ℤ) : 
  (7 / (3 + Real.sqrt 8) = (A * Real.sqrt B + C) / D) →
  (Nat.gcd A.natAbs C.natAbs = 1) →
  (Nat.gcd A.natAbs D.natAbs = 1) →
  (Nat.gcd C.natAbs D.natAbs = 1) →
  A + B + C + D = 23 := by
sorry

end NUMINAMATH_CALUDE_rationalization_sum_l369_36956


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l369_36924

/-- Given a geometric sequence {a_n} where the sum of its first n terms is 2^n - 1,
    this function computes the sum of the first n terms of the sequence {a_n^2}. -/
def sum_of_squares (n : ℕ) : ℚ :=
  (4^n - 1) / 3

/-- The sum of the first n terms of the original geometric sequence {a_n}. -/
def sum_of_original (n : ℕ) : ℕ :=
  2^n - 1

theorem sum_of_squares_theorem (n : ℕ) :
  sum_of_squares n = (4^n - 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l369_36924


namespace NUMINAMATH_CALUDE_period_of_39_over_1428_l369_36926

/-- The period of the repetend of a fraction in binary representation -/
def periodOfRepetend (n d : ℕ) : ℕ := sorry

/-- The order of an element modulo a prime -/
def orderMod (a p : ℕ) : ℕ := sorry

theorem period_of_39_over_1428 :
  let n := 39
  let d := 1428
  let n' := 13
  let d' := 476
  let p₁ := 7
  let p₂ := 17
  (n / d = n' / d') →
  (d' = 2^2 * 119) →
  (119 = p₁ * p₂) →
  (periodOfRepetend n' 119 = periodOfRepetend n' d') →
  periodOfRepetend n d = Nat.lcm (orderMod 2 p₁) (orderMod 2 p₂) := by
  sorry

end NUMINAMATH_CALUDE_period_of_39_over_1428_l369_36926


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l369_36970

theorem sufficient_condition_range (x a : ℝ) :
  (∀ x, (|x - a| < 1 → x^2 + x - 2 > 0) ∧
   ∃ x, x^2 + x - 2 > 0 ∧ |x - a| ≥ 1) →
  a ≤ -3 ∨ a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l369_36970


namespace NUMINAMATH_CALUDE_sugar_for_recipe_l369_36904

/-- The amount of sugar needed for a cake recipe, given the amounts for frosting and cake. -/
theorem sugar_for_recipe (frosting_sugar cake_sugar : ℚ) 
  (h1 : frosting_sugar = 0.6)
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_recipe_l369_36904


namespace NUMINAMATH_CALUDE_blended_number_property_l369_36953

def is_blended_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * a + b

def F (t : ℕ) : ℚ :=
  let t' := (t % 100) * 100 + (t / 100)
  2 * (t + t') / 1111

theorem blended_number_property (p q : ℕ) (a b c d : ℕ) :
  is_blended_number p →
  is_blended_number q →
  p = 1000 * a + 100 * b + 10 * a + b →
  q = 1000 * c + 100 * d + 10 * c + d →
  1 ≤ a →
  a < b →
  b ≤ 9 →
  1 ≤ c →
  c ≤ 9 →
  1 ≤ d →
  d ≤ 9 →
  c ≠ d →
  ∃ (k : ℤ), F p = 17 * k →
  F p + 2 * F q - (4 * a + 3 * b + 2 * d + c) = 0 →
  F (p - q) = 12 ∨ F (p - q) = 16 := by sorry

end NUMINAMATH_CALUDE_blended_number_property_l369_36953


namespace NUMINAMATH_CALUDE_least_faces_triangular_pyramid_l369_36957

structure Shape where
  name : String
  faces : Nat

def triangular_prism : Shape := { name := "Triangular Prism", faces := 5 }
def quadrangular_prism : Shape := { name := "Quadrangular Prism", faces := 6 }
def triangular_pyramid : Shape := { name := "Triangular Pyramid", faces := 4 }
def quadrangular_pyramid : Shape := { name := "Quadrangular Pyramid", faces := 5 }
def truncated_quadrangular_pyramid : Shape := { name := "Truncated Quadrangular Pyramid", faces := 6 }

def shapes : List Shape := [
  triangular_prism,
  quadrangular_prism,
  triangular_pyramid,
  quadrangular_pyramid,
  truncated_quadrangular_pyramid
]

theorem least_faces_triangular_pyramid :
  ∀ s ∈ shapes, triangular_pyramid.faces ≤ s.faces :=
by sorry

end NUMINAMATH_CALUDE_least_faces_triangular_pyramid_l369_36957


namespace NUMINAMATH_CALUDE_inequality_system_solution_l369_36991

theorem inequality_system_solution (x : ℝ) : 
  (2 * x - 1 > 0 ∧ 3 * x > 2 * x + 2) ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l369_36991


namespace NUMINAMATH_CALUDE_coin_found_in_33_moves_l369_36974

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  num_thimbles : Nat
  thimbles_per_move : Nat

/-- Calculates the maximum number of moves needed to guarantee finding the coin. -/
def max_moves_to_find_coin (game : ThimbleGame) : Nat :=
  sorry

/-- Theorem stating that for 100 thimbles and 4 checks per move, 33 moves are sufficient. -/
theorem coin_found_in_33_moves :
  let game : ThimbleGame := { num_thimbles := 100, thimbles_per_move := 4 }
  max_moves_to_find_coin game ≤ 33 := by
  sorry

end NUMINAMATH_CALUDE_coin_found_in_33_moves_l369_36974


namespace NUMINAMATH_CALUDE_complex_modulus_one_l369_36938

theorem complex_modulus_one (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l369_36938


namespace NUMINAMATH_CALUDE_spiral_grid_sum_third_row_l369_36919

/-- Represents a square grid with side length n -/
def Grid (n : ℕ) := Fin n → Fin n → ℕ

/-- Fills the grid in a clockwise spiral starting from the center -/
def fillSpiral (n : ℕ) : Grid n :=
  sorry

/-- Returns the largest number in a given row of the grid -/
def largestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

/-- Returns the smallest number in a given row of the grid -/
def smallestInRow (g : Grid 17) (row : Fin 17) : ℕ :=
  sorry

theorem spiral_grid_sum_third_row :
  let g := fillSpiral 17
  let thirdRow : Fin 17 := 2
  (largestInRow g thirdRow) + (smallestInRow g thirdRow) = 526 :=
by sorry

end NUMINAMATH_CALUDE_spiral_grid_sum_third_row_l369_36919


namespace NUMINAMATH_CALUDE_cubic_equation_has_real_root_l369_36984

theorem cubic_equation_has_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_has_real_root_l369_36984


namespace NUMINAMATH_CALUDE_compare_quadratic_expressions_l369_36994

theorem compare_quadratic_expressions (x : ℝ) : 2*x^2 - 2*x + 1 > x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_compare_quadratic_expressions_l369_36994


namespace NUMINAMATH_CALUDE_inequality_solution_set_l369_36950

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l369_36950


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l369_36982

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l369_36982


namespace NUMINAMATH_CALUDE_factorial_division_l369_36944

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l369_36944


namespace NUMINAMATH_CALUDE_ava_activities_duration_l369_36966

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Represents a duration in hours and minutes -/
structure Duration :=
  (hours : ℕ)
  (minutes : ℕ)

/-- Converts a Duration to total minutes -/
def duration_to_minutes (d : Duration) : ℕ :=
  hours_to_minutes d.hours + d.minutes

/-- The total duration of Ava's activities in minutes -/
def total_duration : ℕ :=
  hours_to_minutes 4 +  -- TV watching
  duration_to_minutes { hours := 2, minutes := 30 } +  -- Video game playing
  duration_to_minutes { hours := 1, minutes := 45 }  -- Walking

theorem ava_activities_duration :
  total_duration = 495 := by sorry

end NUMINAMATH_CALUDE_ava_activities_duration_l369_36966


namespace NUMINAMATH_CALUDE_least_possible_BC_l369_36946

theorem least_possible_BC (AB AC DC BD BC : ℕ) : 
  AB = 7 → 
  AC = 15 → 
  DC = 11 → 
  BD = 25 → 
  BC > AC - AB → 
  BC > BD - DC → 
  BC ≥ 14 ∧ ∀ n : ℕ, (n ≥ 14 → n ≥ BC) → BC = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_least_possible_BC_l369_36946


namespace NUMINAMATH_CALUDE_f_inequality_range_l369_36913

def f (x : ℝ) := -x^3 + 3*x + 2

theorem f_inequality_range (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l369_36913


namespace NUMINAMATH_CALUDE_remainder_of_quadratic_l369_36949

theorem remainder_of_quadratic (a : ℤ) : 
  let n : ℤ := 40 * a + 2
  (n^2 - 3*n + 5) % 40 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_quadratic_l369_36949


namespace NUMINAMATH_CALUDE_vector_operation_l369_36971

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_operation :
  (3 • a - 2 • b) = ![1, 5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l369_36971


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l369_36964

/-- Represents a number in a given base -/
def NumberInBase (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    NumberInBase [5, 8] c = NumberInBase [8, 5] d ∧
    c + d = 15 ∧
    ∀ (c' d' : Nat), c' > 0 → d' > 0 → NumberInBase [5, 8] c' = NumberInBase [8, 5] d' → c' + d' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l369_36964


namespace NUMINAMATH_CALUDE_fencing_calculation_l369_36979

/-- Represents a rectangular field with fencing on three sides -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a given field -/
def required_fencing (field : FencedField) : ℝ :=
  field.length + 2 * field.width

theorem fencing_calculation (field : FencedField) 
  (h1 : field.area = 680)
  (h2 : field.uncovered_side = 34)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 74 := by
  sorry

#check fencing_calculation

end NUMINAMATH_CALUDE_fencing_calculation_l369_36979


namespace NUMINAMATH_CALUDE_total_population_l369_36996

def wildlife_park (num_lions : ℕ) (num_leopards : ℕ) (num_adult_elephants : ℕ) (num_zebras : ℕ) : Prop :=
  (num_lions = 2 * num_leopards) ∧
  (num_adult_elephants = (num_lions * 3 / 4 + num_leopards * 3 / 5) / 2) ∧
  (num_zebras = num_adult_elephants + num_leopards) ∧
  (num_lions = 200)

theorem total_population (num_lions num_leopards num_adult_elephants num_zebras : ℕ) :
  wildlife_park num_lions num_leopards num_adult_elephants num_zebras →
  num_lions + num_leopards + (num_adult_elephants + 100) + num_zebras = 710 :=
by
  sorry

#check total_population

end NUMINAMATH_CALUDE_total_population_l369_36996


namespace NUMINAMATH_CALUDE_largest_angle_of_specific_triangle_l369_36937

/-- Given a triangle with sides 3√2, 6, and 3√10, its largest interior angle is 135°. -/
theorem largest_angle_of_specific_triangle : 
  ∀ (a b c θ : ℝ), 
  a = 3 * Real.sqrt 2 → 
  b = 6 → 
  c = 3 * Real.sqrt 10 → 
  c > a ∧ c > b → 
  θ = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) → 
  θ = 135 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_specific_triangle_l369_36937


namespace NUMINAMATH_CALUDE_value_at_2023_l369_36965

/-- An even function satisfying the given functional equation -/
def EvenFunctionWithProperty (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = 3 - Real.sqrt (6 * f x - f x ^ 2))

/-- The main theorem stating the value of f(2023) -/
theorem value_at_2023 (f : ℝ → ℝ) (h : EvenFunctionWithProperty f) : 
  f 2023 = 3 - (3 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_value_at_2023_l369_36965


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l369_36985

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + c*a = 100) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l369_36985


namespace NUMINAMATH_CALUDE_speedster_convertibles_l369_36942

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 40))  -- 2/3 of total are Speedsters, so 1/3 is 40
  (h2 : 5 * (2 * total / 3) = 4 * total)  -- 4/5 of Speedsters (2/3 of total) are convertibles
  : 4 * total / 5 = 64 := by sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l369_36942
