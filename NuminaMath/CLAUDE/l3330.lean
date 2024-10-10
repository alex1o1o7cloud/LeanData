import Mathlib

namespace tangent_point_on_parabola_l3330_333030

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the derivative of the parabola function
def f' (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_point_on_parabola :
  let M : ℝ × ℝ := (1, 0)
  f M.1 = M.2 ∧ f' M.1 = 3 := by sorry

end tangent_point_on_parabola_l3330_333030


namespace sum_of_squares_of_roots_l3330_333063

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (2 * a^3 - a^2 + 4*a + 10 = 0) → 
  (2 * b^3 - b^2 + 4*b + 10 = 0) → 
  (2 * c^3 - c^2 + 4*c + 10 = 0) → 
  a^2 + b^2 + c^2 = -15/4 := by
  sorry

end sum_of_squares_of_roots_l3330_333063


namespace susie_q_investment_l3330_333054

def pretty_penny_rate : ℝ := 0.03
def five_and_dime_rate : ℝ := 0.05
def total_investment : ℝ := 1000
def total_after_two_years : ℝ := 1090.02
def years : ℕ := 2

theorem susie_q_investment (x : ℝ) :
  x * (1 + pretty_penny_rate) ^ years + (total_investment - x) * (1 + five_and_dime_rate) ^ years = total_after_two_years →
  x = 300 := by
sorry

end susie_q_investment_l3330_333054


namespace discarded_numbers_l3330_333003

-- Define the set of numbers
def numbers : Finset ℕ := Finset.range 11 \ {0}

-- Define the type for a distribution on a rectangular block
structure BlockDistribution where
  vertices : Finset ℕ
  face_sum : ℕ
  is_valid : vertices ⊆ numbers ∧ vertices.card = 8 ∧ face_sum = 18

-- Theorem statement
theorem discarded_numbers (d : BlockDistribution) :
  numbers \ d.vertices = {9, 10} := by
  sorry

end discarded_numbers_l3330_333003


namespace regular_polygon_properties_l3330_333033

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides and interior angles of 150°. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (perimeter side_length : ℝ) (interior_angle : ℝ),
    perimeter = 180 →
    side_length = 15 →
    n * side_length = perimeter →
    interior_angle = (n - 2) * 180 / n →
    n = 12 ∧ interior_angle = 150 := by
  sorry

end regular_polygon_properties_l3330_333033


namespace grid_shading_l3330_333065

/-- Given a 4 × 5 grid with 3 squares already shaded, 
    prove that 7 additional squares need to be shaded 
    to have half of all squares shaded. -/
theorem grid_shading (grid_width : Nat) (grid_height : Nat) 
  (total_squares : Nat) (already_shaded : Nat) (half_squares : Nat) 
  (additional_squares : Nat) : 
  grid_width = 4 → 
  grid_height = 5 → 
  total_squares = grid_width * grid_height →
  already_shaded = 3 →
  half_squares = total_squares / 2 →
  additional_squares = half_squares - already_shaded →
  additional_squares = 7 := by
sorry


end grid_shading_l3330_333065


namespace correct_calculation_l3330_333046

theorem correct_calculation (x : ℝ) : 
  x / 3.6 = 2.5 → (x * 3.6) / 2 = 16.2 := by
  sorry

end correct_calculation_l3330_333046


namespace log_division_simplification_l3330_333037

theorem log_division_simplification : 
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end log_division_simplification_l3330_333037


namespace perfect_square_condition_l3330_333010

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + k)^2) → (m = 6 ∨ m = -6) :=
by sorry

end perfect_square_condition_l3330_333010


namespace bus_express_speed_l3330_333049

/-- Proves that the speed of a bus in express mode is 48 km/h given specific conditions -/
theorem bus_express_speed (route_length : ℝ) (time_reduction : ℝ) (speed_increase : ℝ)
  (h1 : route_length = 16)
  (h2 : time_reduction = 1 / 15)
  (h3 : speed_increase = 8)
  : ∃ x : ℝ, x = 48 ∧ 
    route_length / (x - speed_increase) - route_length / x = time_reduction :=
by sorry

end bus_express_speed_l3330_333049


namespace elder_age_l3330_333007

/-- The age difference between two people -/
def age_difference : ℕ := 20

/-- The number of years ago when the elder was 5 times as old as the younger -/
def years_ago : ℕ := 8

/-- The ratio of elder's age to younger's age in the past -/
def age_ratio : ℕ := 5

theorem elder_age (younger_age elder_age : ℕ) : 
  (elder_age = younger_age + age_difference) → 
  (elder_age - years_ago = age_ratio * (younger_age - years_ago)) →
  elder_age = 33 := by
sorry

end elder_age_l3330_333007


namespace quadratic_equation_1_l3330_333006

theorem quadratic_equation_1 (x : ℝ) : x^2 + 16 = 8*x → x = 4 := by
  sorry

end quadratic_equation_1_l3330_333006


namespace units_digit_problem_l3330_333055

theorem units_digit_problem : ∃ n : ℕ, n < 10 ∧ 
  (72^129 + 36^93 + 57^73 - 45^105) % 10 = n ∧ n = 0 := by
  sorry

end units_digit_problem_l3330_333055


namespace book_purchase_solution_l3330_333059

/-- Represents the cost and purchase details of two types of books -/
structure BookPurchase where
  costA : ℕ  -- Cost of book A
  costB : ℕ  -- Cost of book B
  totalBooks : ℕ  -- Total number of books to purchase
  maxCost : ℕ  -- Maximum total cost

/-- Defines the conditions of the book purchase problem -/
def validBookPurchase (bp : BookPurchase) : Prop :=
  bp.costB = bp.costA + 20 ∧  -- Condition 1
  540 / bp.costA = 780 / bp.costB ∧  -- Condition 2
  bp.totalBooks = 70 ∧  -- Condition 3
  bp.maxCost = 3550  -- Condition 4

/-- Theorem stating the solution to the book purchase problem -/
theorem book_purchase_solution (bp : BookPurchase) 
  (h : validBookPurchase bp) : 
  bp.costA = 45 ∧ bp.costB = 65 ∧ 
  (∀ m : ℕ, m * bp.costA + (bp.totalBooks - m) * bp.costB ≤ bp.maxCost → m ≥ 50) :=
sorry

end book_purchase_solution_l3330_333059


namespace positive_real_inequality_l3330_333060

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end positive_real_inequality_l3330_333060


namespace total_tiles_to_replace_l3330_333082

/-- Represents the layout of paths in the park -/
structure ParkPaths where
  horizontalLengths : List Nat
  verticalLengths : List Nat

/-- Calculates the total number of tiles needed for replacement -/
def totalTiles (paths : ParkPaths) : Nat :=
  let horizontalSum := paths.horizontalLengths.sum
  let verticalSum := paths.verticalLengths.sum
  let intersections := 16  -- This value is derived from the problem description
  horizontalSum + verticalSum - intersections

/-- The main theorem stating the total number of tiles to be replaced -/
theorem total_tiles_to_replace :
  ∃ (paths : ParkPaths),
    paths.horizontalLengths = [30, 50, 30, 20, 20, 50] ∧
    paths.verticalLengths = [20, 50, 20, 50, 50] ∧
    totalTiles paths = 374 :=
  sorry

end total_tiles_to_replace_l3330_333082


namespace inequality_proof_l3330_333028

/-- Given f(x) = e^x - x^2, prove that for all x > 0, (e^x + (2-e)x - 1) / x ≥ ln x + 1 -/
theorem inequality_proof (x : ℝ) (hx : x > 0) : (Real.exp x + (2 - Real.exp 1) * x - 1) / x ≥ Real.log x + 1 := by
  sorry

end inequality_proof_l3330_333028


namespace expression_evaluation_l3330_333048

theorem expression_evaluation :
  let a : ℚ := 4/3
  (7 * a^2 - 15 * a + 2) * (3 * a - 4) = 0 := by sorry

end expression_evaluation_l3330_333048


namespace stratified_sampling_c_l3330_333062

/-- Represents the number of individuals in each sample -/
structure SampleSizes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The ratio of individuals in samples A, B, and C -/
def sample_ratio : SampleSizes := { A := 5, B := 3, C := 2 }

/-- The total sample size for stratified sampling -/
def total_sample_size : ℕ := 100

/-- Calculates the number of individuals to be drawn from a specific sample -/
def stratified_sample_size (ratio : ℕ) : ℕ :=
  (total_sample_size * ratio) / (sample_ratio.A + sample_ratio.B + sample_ratio.C)

theorem stratified_sampling_c :
  stratified_sample_size sample_ratio.C = 20 := by
  sorry

end stratified_sampling_c_l3330_333062


namespace root_ratio_theorem_l3330_333080

theorem root_ratio_theorem (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + k*x₁ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₂^2 + k*x₂ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₁ ≠ x₂ →
  x₁^2020 / x₂^2021 = -2/3 := by
sorry

end root_ratio_theorem_l3330_333080


namespace orange_juice_percentage_l3330_333064

/-- Represents the juice extraction information for a fruit type -/
structure FruitJuice where
  fruitCount : ℕ
  juiceAmount : ℕ

/-- Represents the blend composition -/
structure Blend where
  pearCount : ℕ
  orangeCount : ℕ

def calculateJuicePercentage (pearJuice : FruitJuice) (orangeJuice : FruitJuice) (blend : Blend) : ℚ :=
  let pearJuiceRate := pearJuice.juiceAmount / pearJuice.fruitCount
  let orangeJuiceRate := orangeJuice.juiceAmount / orangeJuice.fruitCount
  let totalPearJuice := pearJuiceRate * blend.pearCount
  let totalOrangeJuice := orangeJuiceRate * blend.orangeCount
  let totalJuice := totalPearJuice + totalOrangeJuice
  totalOrangeJuice / totalJuice

theorem orange_juice_percentage
  (pearJuice : FruitJuice)
  (orangeJuice : FruitJuice)
  (blend : Blend)
  (h1 : pearJuice.fruitCount = 5 ∧ pearJuice.juiceAmount = 10)
  (h2 : orangeJuice.fruitCount = 4 ∧ orangeJuice.juiceAmount = 12)
  (h3 : blend.pearCount = 9 ∧ blend.orangeCount = 6) :
  calculateJuicePercentage pearJuice orangeJuice blend = 1/2 := by
  sorry

#eval calculateJuicePercentage ⟨5, 10⟩ ⟨4, 12⟩ ⟨9, 6⟩

end orange_juice_percentage_l3330_333064


namespace elastic_collision_mass_and_velocity_ratios_l3330_333072

/-- Represents the masses and velocities in an elastic collision -/
structure CollisionSystem where
  m₁ : ℝ
  m₂ : ℝ
  v₀ : ℝ
  v₁ : ℝ
  v₂ : ℝ

/-- Conditions for the elastic collision system -/
def ElasticCollision (s : CollisionSystem) : Prop :=
  s.m₁ > 0 ∧ s.m₂ > 0 ∧ s.v₀ > 0 ∧ s.v₁ > 0 ∧ s.v₂ > 0 ∧
  s.v₂ = 4 * s.v₁ ∧
  s.m₁ * s.v₀ = s.m₁ * s.v₁ + s.m₂ * s.v₂ ∧
  s.m₁ * s.v₀^2 = s.m₁ * s.v₁^2 + s.m₂ * s.v₂^2

theorem elastic_collision_mass_and_velocity_ratios (s : CollisionSystem) 
  (h : ElasticCollision s) : s.m₂ / s.m₁ = 1/2 ∧ s.v₀ / s.v₁ = 3 := by
  sorry

end elastic_collision_mass_and_velocity_ratios_l3330_333072


namespace stating_safe_zone_condition_l3330_333013

/-- Represents the fuse burning speed in cm/s -/
def fuse_speed : ℝ := 0.5

/-- Represents the person's running speed in m/s -/
def person_speed : ℝ := 4

/-- Represents the safe zone distance in meters -/
def safe_distance : ℝ := 150

/-- 
Theorem stating the condition for a person to reach the safe zone before the fuse burns out.
x represents the fuse length in cm.
-/
theorem safe_zone_condition (x : ℝ) :
  (x ≥ 0) →
  (person_speed * (x / fuse_speed) ≥ safe_distance) ↔
  (4 * (x / 0.5) ≥ 150) :=
sorry

end stating_safe_zone_condition_l3330_333013


namespace amy_book_count_l3330_333098

theorem amy_book_count (maddie_books luisa_books : ℕ) 
  (h1 : maddie_books = 15)
  (h2 : luisa_books = 18)
  (h3 : luisa_books + amy_books = maddie_books + 9) : 
  amy_books = 6 := by
  sorry

end amy_book_count_l3330_333098


namespace min_red_chips_l3330_333019

theorem min_red_chips (w b r : ℕ) : 
  b ≥ (3 * w) / 4 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  r ≥ 132 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ (3 * w') / 4 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 132 :=
by sorry

end min_red_chips_l3330_333019


namespace classroom_writing_instruments_l3330_333042

theorem classroom_writing_instruments :
  let total_bags : ℕ := 16
  let compartments_per_bag : ℕ := 6
  let max_instruments_per_compartment : ℕ := 8
  let empty_compartments : ℕ := 5
  let partially_filled_compartment : ℕ := 1
  let instruments_in_partially_filled : ℕ := 6
  
  let total_compartments : ℕ := total_bags * compartments_per_bag
  let filled_compartments : ℕ := total_compartments - empty_compartments - partially_filled_compartment
  
  let total_instruments : ℕ := 
    filled_compartments * max_instruments_per_compartment + 
    partially_filled_compartment * instruments_in_partially_filled
  
  total_instruments = 726 := by
  sorry

end classroom_writing_instruments_l3330_333042


namespace number_is_two_l3330_333096

theorem number_is_two (x y : ℝ) (n : ℝ) 
  (h1 : n * (x - y) = 4)
  (h2 : 6 * x - 3 * y = 12) : n = 2 := by
  sorry

end number_is_two_l3330_333096


namespace sum_of_reciprocals_squared_l3330_333036

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560/83521 := by
  sorry

end sum_of_reciprocals_squared_l3330_333036


namespace product_of_logs_l3330_333027

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 1560) →
  (Real.log b / Real.log a = 3) →
  (a + b : ℕ) = 1740 := by sorry

end product_of_logs_l3330_333027


namespace petes_flag_total_shapes_l3330_333073

/-- The number of stars on the US flag -/
def us_stars : ℕ := 50

/-- The number of stripes on the US flag -/
def us_stripes : ℕ := 13

/-- The number of circles on Pete's flag -/
def petes_circles : ℕ := us_stars / 2 - 3

/-- The number of squares on Pete's flag -/
def petes_squares : ℕ := us_stripes * 2 + 6

/-- Theorem stating the total number of shapes on Pete's flag -/
theorem petes_flag_total_shapes :
  petes_circles + petes_squares = 54 := by sorry

end petes_flag_total_shapes_l3330_333073


namespace regression_lines_intersect_at_mean_l3330_333089

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def point_on_line (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Theorem: Two different regression lines for the same dataset intersect at the sample mean -/
theorem regression_lines_intersect_at_mean 
  (l₁ l₂ : RegressionLine) 
  (x_mean y_mean : ℝ) 
  (h_different : l₁ ≠ l₂) 
  (h_on_l₁ : point_on_line l₁ x_mean y_mean)
  (h_on_l₂ : point_on_line l₂ x_mean y_mean) : 
  ∃ (x y : ℝ), x = x_mean ∧ y = y_mean ∧ point_on_line l₁ x y ∧ point_on_line l₂ x y :=
sorry

end regression_lines_intersect_at_mean_l3330_333089


namespace max_min_f_l3330_333040

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- Define the interval
def a : ℝ := 0
def b : ℝ := 3

-- Theorem statement
theorem max_min_f :
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ ∀ (y : ℝ), y ∈ Set.Icc a b → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≤ 5) ∧
  (∀ (x : ℝ), x ∈ Set.Icc a b → f x ≥ -15) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = 5) ∧
  (∃ (x : ℝ), x ∈ Set.Icc a b ∧ f x = -15) :=
by sorry

end max_min_f_l3330_333040


namespace coat_price_proof_l3330_333009

theorem coat_price_proof (reduction : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  reduction = 400 →
  percentage = 0.8 →
  percentage * original_price = reduction →
  original_price = 500 := by
sorry

end coat_price_proof_l3330_333009


namespace whack_a_mole_tickets_whack_a_mole_tickets_proof_l3330_333074

theorem whack_a_mole_tickets : ℕ → Prop :=
  fun whack_tickets =>
    let skee_tickets : ℕ := 9
    let candy_cost : ℕ := 6
    let candies_bought : ℕ := 7
    whack_tickets + skee_tickets = candy_cost * candies_bought →
    whack_tickets = 33

-- The proof is omitted
theorem whack_a_mole_tickets_proof : whack_a_mole_tickets 33 := by
  sorry

end whack_a_mole_tickets_whack_a_mole_tickets_proof_l3330_333074


namespace special_right_triangle_hypotenuse_l3330_333052

/-- A right triangle with specific leg relationship and area -/
structure SpecialRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  leg_relationship : longer_leg = 3 * shorter_leg - 3
  area_condition : (1 / 2) * shorter_leg * longer_leg = 108
  right_triangle : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The hypotenuse of the special right triangle is √657 -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) : t.hypotenuse = Real.sqrt 657 := by
  sorry

end special_right_triangle_hypotenuse_l3330_333052


namespace student_attendance_probability_l3330_333092

theorem student_attendance_probability :
  let p_absent : ℝ := 1 / 20
  let p_present : ℝ := 1 - p_absent
  let p_one_absent_one_present : ℝ := p_absent * p_present + p_present * p_absent
  p_one_absent_one_present = 0.095 := by
  sorry

end student_attendance_probability_l3330_333092


namespace solution_set_l3330_333058

theorem solution_set (x : ℝ) : 
  (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1/4 ∧ x - 2 > 0 → x > 2 := by
  sorry

end solution_set_l3330_333058


namespace salary_spending_percentage_l3330_333012

theorem salary_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_rate : ℝ) 
  (h1 : total_salary = 5000)
  (h2 : a_salary = 3750)
  (h3 : b_spending_rate = 0.85)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  a_spending_rate = 0.95 := by
sorry

end salary_spending_percentage_l3330_333012


namespace cubic_equation_solution_l3330_333022

theorem cubic_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = 999 ↔ (x = 12 ∧ y = 9) ∨ (x = 10 ∧ y = 1) :=
by sorry

end cubic_equation_solution_l3330_333022


namespace min_value_of_sum_of_inverse_trig_functions_l3330_333094

theorem min_value_of_sum_of_inverse_trig_functions
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (m : ℝ), ∀ (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2),
    m ≤ a / (Real.sin θ)^3 + b / (Real.cos θ)^3 ∧
    m = (a^(2/5) + b^(2/5))^(5/2) := by
  sorry

end min_value_of_sum_of_inverse_trig_functions_l3330_333094


namespace log_inequality_l3330_333029

theorem log_inequality (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x) / (x + 1) + 1 / x > (Real.log x) / (x - 1) := by
  sorry

end log_inequality_l3330_333029


namespace radiator_antifreeze_percentage_l3330_333021

/-- The capacity of the radiator in liters -/
def radiator_capacity : ℝ := 6

/-- The volume of liquid replaced with pure antifreeze in liters -/
def replaced_volume : ℝ := 1

/-- The final percentage of antifreeze in the mixture -/
def final_percentage : ℝ := 0.5

/-- The initial percentage of antifreeze in the radiator -/
def initial_percentage : ℝ := 0.4

theorem radiator_antifreeze_percentage :
  let remaining_volume := radiator_capacity - replaced_volume
  let initial_antifreeze := initial_percentage * radiator_capacity
  let remaining_antifreeze := initial_antifreeze - initial_percentage * replaced_volume
  let final_antifreeze := remaining_antifreeze + replaced_volume
  final_antifreeze = final_percentage * radiator_capacity :=
by sorry

end radiator_antifreeze_percentage_l3330_333021


namespace complex_modulus_l3330_333001

theorem complex_modulus (z : ℂ) : (1 + I) * z = 2 * I → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l3330_333001


namespace geometric_sequence_max_point_l3330_333015

/-- Given real numbers a, b, c, and d forming a geometric sequence,
    and (b, c) being the coordinates of the maximum point of the curve y = 3x - x^3,
    prove that ad = 2. -/
theorem geometric_sequence_max_point (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →  -- maximum point condition
  c = 3 * b - b^3 →  -- y-coordinate of maximum point
  a * d = 2 := by
  sorry

end geometric_sequence_max_point_l3330_333015


namespace fraction_meaningful_l3330_333025

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end fraction_meaningful_l3330_333025


namespace marias_car_trip_l3330_333095

theorem marias_car_trip (D : ℝ) : 
  (D / 2 / 4 / 3 + D / 2 / 4 * 2 / 3 + D / 2 * 3 / 4) = 630 → D = 840 := by
  sorry

end marias_car_trip_l3330_333095


namespace remainder_8927_mod_11_l3330_333057

theorem remainder_8927_mod_11 : 8927 % 11 = 8 := by
  sorry

end remainder_8927_mod_11_l3330_333057


namespace complex_number_properties_l3330_333084

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end complex_number_properties_l3330_333084


namespace last_card_is_diamond_six_l3330_333069

/-- Represents a playing card --/
inductive Card
| Joker : Bool → Card  -- True for Big Joker, False for Little Joker
| Number : Nat → Suit → Card
| Face : Face → Suit → Card

/-- Represents the suit of a card --/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- Represents face cards --/
inductive Face
| Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of 54 cards in the specified order --/
def standardDeck : Deck := sorry

/-- Combines two decks --/
def combinedDeck (d1 d2 : Deck) : Deck := sorry

/-- Applies the discard-and-place rule to a deck --/
def applyRule (d : Deck) : Card := sorry

/-- Theorem: The last remaining card is the Diamond 6 --/
theorem last_card_is_diamond_six :
  let d1 := standardDeck
  let d2 := standardDeck
  let combined := combinedDeck d1 d2
  applyRule combined = Card.Number 6 Suit.Diamonds := by sorry

end last_card_is_diamond_six_l3330_333069


namespace num_pupils_correct_l3330_333093

/-- The number of pupils sent up for examination -/
def num_pupils : ℕ := 21

/-- The average marks of all pupils -/
def average_marks : ℚ := 39

/-- The marks of the 4 specific pupils -/
def specific_pupils_marks : List ℕ := [25, 12, 15, 19]

/-- The average marks if the 4 specific pupils were removed -/
def average_without_specific : ℚ := 44

/-- Theorem stating that the number of pupils is correct given the conditions -/
theorem num_pupils_correct :
  (average_marks * num_pupils : ℚ) =
  (average_without_specific * (num_pupils - 4) : ℚ) + (specific_pupils_marks.sum : ℚ) :=
by sorry

end num_pupils_correct_l3330_333093


namespace regular_decagon_perimeter_l3330_333083

/-- A regular decagon is a polygon with 10 sides of equal length -/
def RegularDecagon := Nat

/-- The side length of a regular decagon -/
def sideLength (d : RegularDecagon) : ℝ := 3

/-- The perimeter of a regular decagon -/
def perimeter (d : RegularDecagon) : ℝ := 10 * sideLength d

theorem regular_decagon_perimeter (d : RegularDecagon) : 
  perimeter d = 30 := by
  sorry

end regular_decagon_perimeter_l3330_333083


namespace intersection_of_M_and_N_l3330_333067

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l3330_333067


namespace quadratic_inequality_range_l3330_333032

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ m ∈ Set.Ioc (-4) 0 := by
  sorry

end quadratic_inequality_range_l3330_333032


namespace students_in_both_math_and_science_l3330_333053

theorem students_in_both_math_and_science 
  (total : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total = 40) 
  (h2 : not_math = 10) 
  (h3 : not_science = 15) 
  (h4 : not_either = 2) : 
  total - not_math + total - not_science - (total - not_either) = 17 := by
sorry

end students_in_both_math_and_science_l3330_333053


namespace problem_statement_l3330_333068

theorem problem_statement (a b c d : ℝ) : 
  (Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3) →
  (a - b + c - d = -7) := by
sorry

end problem_statement_l3330_333068


namespace right_triangle_trig_l3330_333000

theorem right_triangle_trig (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (∀ S, S ≠ R → (Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2))^2 = (S.1 - R.1)^2 + (S.2 - R.2)^2) →
  pq = 15 →
  pr = 9 →
  qr^2 + pr^2 = pq^2 →
  (qr / pq) = 4/5 ∧ (pr / pq) = 3/5 :=
by sorry

end right_triangle_trig_l3330_333000


namespace sequence_term_correct_l3330_333099

def sequence_sum (n : ℕ) : ℕ := 3 + 2^n

def sequence_term (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n-1)

theorem sequence_term_correct :
  ∀ n : ℕ, n ≥ 1 →
  sequence_sum n - sequence_sum (n-1) = sequence_term n :=
sorry

end sequence_term_correct_l3330_333099


namespace smallest_b_in_arithmetic_geometric_progression_l3330_333075

theorem smallest_b_in_arithmetic_geometric_progression (a b c : ℤ) : 
  a < c → c < b → 
  (2 * c = a + b) →  -- arithmetic progression condition
  (b * b = a * c) →  -- geometric progression condition
  (∀ b' : ℤ, (∃ a' c' : ℤ, a' < c' ∧ c' < b' ∧ 
    (2 * c' = a' + b') ∧ 
    (b' * b' = a' * c')) → b' ≥ 2) →
  b = 2 :=
by sorry

end smallest_b_in_arithmetic_geometric_progression_l3330_333075


namespace playground_count_l3330_333011

/-- The total number of people on the playground after late arrivals -/
def total_people (initial_boys initial_girls teachers late_boys late_girls : ℕ) : ℕ :=
  initial_boys + initial_girls + teachers + late_boys + late_girls

/-- Theorem stating the total number of people on the playground after late arrivals -/
theorem playground_count : total_people 44 53 5 3 2 = 107 := by
  sorry

end playground_count_l3330_333011


namespace a_range_theorem_l3330_333031

-- Define the type for real numbers greater than zero
def PositiveReal := {x : ℝ // x > 0}

-- Define the monotonically increasing property for a^x
def MonotonicallyIncreasing (a : PositiveReal) : Prop :=
  ∀ x y : ℝ, x < y → (a.val : ℝ) ^ x < (a.val : ℝ) ^ y

-- Define the property that x^2 - ax + 1 > 0 does not hold for all x
def NotAlwaysPositive (a : PositiveReal) : Prop :=
  ¬(∀ x : ℝ, x^2 - (a.val : ℝ) * x + 1 > 0)

-- State the theorem
theorem a_range_theorem (a : PositiveReal) 
  (h1 : MonotonicallyIncreasing a) 
  (h2 : NotAlwaysPositive a) : 
  (a.val : ℝ) ≥ 2 :=
sorry

end a_range_theorem_l3330_333031


namespace zongzi_sales_and_profit_l3330_333081

/-- The daily sales volume function for zongzi -/
def sales_volume (x : ℝ) : ℝ := 800 * x + 400

/-- The maximum daily production of zongzi -/
def max_production : ℝ := 1100

/-- The initial profit per zongzi in yuan -/
def initial_profit : ℝ := 2

/-- The total profit function for zongzi sales -/
def total_profit (x : ℝ) : ℝ := (initial_profit - x) * sales_volume x

theorem zongzi_sales_and_profit :
  (sales_volume 0.2 = 560) ∧ 
  (total_profit 0.2 = 1008) ∧
  (∃ x : ℝ, total_profit x = 1200 ∧ x = 0.5 ∧ sales_volume x ≤ max_production) :=
by sorry

end zongzi_sales_and_profit_l3330_333081


namespace abc_zero_l3330_333038

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end abc_zero_l3330_333038


namespace diophantine_equation_solution_l3330_333066

theorem diophantine_equation_solution (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end diophantine_equation_solution_l3330_333066


namespace remainder_11_power_2023_mod_19_l3330_333017

theorem remainder_11_power_2023_mod_19 : 11^2023 % 19 = 17 := by
  sorry

end remainder_11_power_2023_mod_19_l3330_333017


namespace no_real_solutions_l3330_333043

theorem no_real_solutions (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) ↔ m = -5 := by
  sorry

end no_real_solutions_l3330_333043


namespace reflected_light_is_two_thirds_l3330_333024

/-- A mirror that reflects half the light shined on it back and passes the other half onward -/
structure FiftyPercentMirror :=
  (reflect : ℝ → ℝ)
  (pass : ℝ → ℝ)
  (reflect_half : ∀ x, reflect x = x / 2)
  (pass_half : ∀ x, pass x = x / 2)

/-- Two parallel fifty percent mirrors -/
structure TwoParallelMirrors :=
  (mirror1 : FiftyPercentMirror)
  (mirror2 : FiftyPercentMirror)

/-- The fraction of light reflected back to the left by two parallel fifty percent mirrors -/
def reflected_light (mirrors : TwoParallelMirrors) (initial_light : ℝ) : ℝ :=
  sorry

/-- Theorem: The total fraction of light reflected back to the left by two parallel "fifty percent mirrors" is 2/3 when light is shined from the left -/
theorem reflected_light_is_two_thirds (mirrors : TwoParallelMirrors) (initial_light : ℝ) :
  reflected_light mirrors initial_light = 2/3 * initial_light :=
sorry

end reflected_light_is_two_thirds_l3330_333024


namespace base_representation_of_200_l3330_333014

theorem base_representation_of_200 :
  ∃! b : ℕ, b > 1 ∧ b^5 ≤ 200 ∧ 200 < b^6 := by sorry

end base_representation_of_200_l3330_333014


namespace system_solution_l3330_333045

theorem system_solution : ∃ (x y : ℝ), (3 * x = -9 - 3 * y) ∧ (2 * x = 3 * y - 22) := by
  use -5, 2
  sorry

#check system_solution

end system_solution_l3330_333045


namespace turnips_sum_l3330_333061

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := 9

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ := keith_turnips + alyssa_turnips

theorem turnips_sum :
  total_turnips = 15 := by sorry

end turnips_sum_l3330_333061


namespace square_area_from_diagonal_l3330_333051

theorem square_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * s = x * x / 2 :=
by
  sorry

#check square_area_from_diagonal

end square_area_from_diagonal_l3330_333051


namespace equation_solution_l3330_333041

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 105) / 24 ∧ 
     x₂ = (3 - Real.sqrt 105) / 24) ∧ 
    (∀ x : ℝ, 4 * (3 * x)^2 + 2 * (3 * x) + 7 = 3 * (8 * x^2 + 3 * x + 3) ↔ 
      x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l3330_333041


namespace expand_polynomial_l3330_333076

theorem expand_polynomial (x : ℝ) : (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 := by
  sorry

end expand_polynomial_l3330_333076


namespace balance_problem_l3330_333035

/-- The problem of balancing weights on a scale --/
theorem balance_problem :
  let total_weight : ℝ := 4.5 -- in kg
  let num_weights : ℕ := 9
  let weight_per_item : ℝ := total_weight / num_weights -- in kg
  let pencil_case_weight : ℝ := 0.85 -- in kg
  let dictionary_weight : ℝ := 1.05 -- in kg
  let num_weights_on_scale : ℕ := 2
  let num_dictionaries : ℕ := 5
  ∃ (num_pencil_cases : ℕ),
    (num_weights_on_scale * weight_per_item + num_pencil_cases * pencil_case_weight) =
    (num_dictionaries * dictionary_weight) ∧
    num_pencil_cases = 5 :=
by sorry

end balance_problem_l3330_333035


namespace bus_train_speed_ratio_l3330_333008

/-- The fraction of the speed of a bus compared to the speed of a train -/
theorem bus_train_speed_ratio :
  -- The ratio between the speed of a train and a car
  ∀ (train_speed car_speed : ℝ),
  train_speed / car_speed = 16 / 15 →
  -- A bus covered 320 km in 5 hours
  ∀ (bus_speed : ℝ),
  bus_speed * 5 = 320 →
  -- The car will cover 525 km in 7 hours
  car_speed * 7 = 525 →
  -- The fraction of the speed of the bus compared to the speed of the train
  bus_speed / train_speed = 4 / 5 := by
sorry

end bus_train_speed_ratio_l3330_333008


namespace min_value_of_xy_l3330_333050

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1/2) :
  x * y ≥ 16 := by
  sorry

end min_value_of_xy_l3330_333050


namespace difference_of_squares_example_l3330_333005

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end difference_of_squares_example_l3330_333005


namespace system_solution_l3330_333044

theorem system_solution (x y : ℝ) (h1 : x + 2*y = 8) (h2 : 2*x + y = 7) : x + y = 5 := by
  sorry

end system_solution_l3330_333044


namespace ana_beto_game_l3330_333086

def is_valid_sequence (seq : List Int) : Prop :=
  seq.length = 2016 ∧ (seq.count 1 = 1008) ∧ (seq.count (-1) = 1008)

def block_sum_squares (blocks : List (List Int)) : Int :=
  (blocks.map (λ block => (block.sum)^2)).sum

theorem ana_beto_game (N : Nat) :
  (∃ (seq : List Int) (blocks : List (List Int)),
    is_valid_sequence seq ∧
    seq = blocks.join ∧
    block_sum_squares blocks = N) ↔
  (N % 2 = 0 ∧ N ≤ 2016) :=
sorry

end ana_beto_game_l3330_333086


namespace last_digit_square_periodicity_and_symmetry_l3330_333090

theorem last_digit_square_periodicity_and_symmetry :
  ∀ (n : ℕ), 
    (n^2 % 10 = ((n + 10)^2) % 10) ∧
    (∀ (k : ℕ), k ≤ 4 → (k^2 % 10 = ((10 - k)^2) % 10)) :=
by sorry

end last_digit_square_periodicity_and_symmetry_l3330_333090


namespace keith_attended_four_games_l3330_333070

/-- The number of football games Keith attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

theorem keith_attended_four_games :
  let total_games : ℕ := 8
  let missed_games : ℕ := 4
  games_attended total_games missed_games = 4 := by
  sorry

end keith_attended_four_games_l3330_333070


namespace license_plate_theorem_l3330_333004

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible digits -/
def digit_count : ℕ := 10

/-- The number of license plate combinations -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count * vowel_count * digit_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end license_plate_theorem_l3330_333004


namespace odd_digits_346_base5_l3330_333039

/-- Counts the number of odd digits in a base-5 number --/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 --/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_346_base5 : 
  countOddDigitsBase5 (toBase5 346) = 2 := by sorry

end odd_digits_346_base5_l3330_333039


namespace g_of_4_l3330_333047

def g (x : ℝ) : ℝ := 5 * x + 2

theorem g_of_4 : g 4 = 22 := by sorry

end g_of_4_l3330_333047


namespace integer_solutions_quadratic_equation_l3330_333034

theorem integer_solutions_quadratic_equation :
  ∀ m n : ℤ, n^2 - 3*m*n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) := by
sorry

end integer_solutions_quadratic_equation_l3330_333034


namespace square_ratios_l3330_333097

theorem square_ratios (a b : ℝ) (h : b = 3 * a) :
  (4 * b) / (4 * a) = 3 ∧ (b * b) / (a * a) = 9 := by
  sorry

end square_ratios_l3330_333097


namespace trajectory_equation_l3330_333020

/-- The equation of the trajectory of the center of a circle that passes through point A (2, 0) and is tangent to the circle x^2 + 4x + y^2 - 32 = 0 is x^2/9 + y^2/5 = 1 -/
theorem trajectory_equation : ∃ (f : ℝ × ℝ → ℝ), 
  (∀ (x y : ℝ), f (x, y) = 0 ↔ x^2/9 + y^2/5 = 1) ∧
  (∀ (x y : ℝ), f (x, y) = 0 → 
    ∃ (r : ℝ), r > 0 ∧
    (∀ (u v : ℝ), (u - x)^2 + (v - y)^2 = r^2 → 
      ((u - 2)^2 + v^2 = 0 ∨ u^2 + 4*u + v^2 - 32 = 0))) :=
sorry

end trajectory_equation_l3330_333020


namespace smallest_number_l3330_333018

theorem smallest_number (a b c d : ℝ) (h1 : a = 2) (h2 : b = -2.5) (h3 : c = 0) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end smallest_number_l3330_333018


namespace remainder_8354_mod_11_l3330_333088

theorem remainder_8354_mod_11 : 8354 % 11 = 6 := by
  sorry

end remainder_8354_mod_11_l3330_333088


namespace baker_productivity_l3330_333071

/-- The number of ovens the baker has -/
def num_ovens : ℕ := 4

/-- The number of hours the baker bakes on weekdays -/
def weekday_hours : ℕ := 5

/-- The number of hours the baker bakes on weekend days -/
def weekend_hours : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks the baker bakes -/
def num_weeks : ℕ := 3

/-- The total number of loaves baked in 3 weeks -/
def total_loaves : ℕ := 1740

/-- The number of loaves baked per hour in one oven -/
def loaves_per_hour : ℚ :=
  total_loaves / (num_ovens * (weekdays * weekday_hours + weekend_days * weekend_hours) * num_weeks)

theorem baker_productivity : loaves_per_hour = 5 := by
  sorry

end baker_productivity_l3330_333071


namespace subset_P_l3330_333002

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end subset_P_l3330_333002


namespace max_product_range_l3330_333085

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_range (h k : ℝ → ℝ) 
  (h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 5)
  (k_range : ∀ x, 0 ≤ k x ∧ k x ≤ 4) :
  ∃ d, ∀ x, h x ^ 2 * k x ≤ d ∧ d = 100 :=
sorry

end max_product_range_l3330_333085


namespace crow_worm_consumption_l3330_333087

/-- Given that 3 crows eat 30 worms in one hour, prove that 5 crows will eat 100 worms in 2 hours. -/
theorem crow_worm_consumption (crows_per_hour : ℕ → ℕ → ℕ) : 
  crows_per_hour 3 30 = 1  -- 3 crows eat 30 worms in 1 hour
  → crows_per_hour 5 100 = 2  -- 5 crows eat 100 worms in 2 hours
:= by sorry

end crow_worm_consumption_l3330_333087


namespace marbles_given_to_joan_l3330_333026

theorem marbles_given_to_joan (initial_marbles : ℝ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 9.0) 
  (h2 : remaining_marbles = 6) :
  initial_marbles - remaining_marbles = 3 := by
  sorry

end marbles_given_to_joan_l3330_333026


namespace min_value_on_circle_l3330_333056

theorem min_value_on_circle (x y : ℝ) :
  (x - 2)^2 + (y - 3)^2 = 1 →
  ∃ (z : ℝ), z = 14 - 2 * Real.sqrt 13 ∧ ∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → x^2 + y^2 ≤ a^2 + b^2 :=
by sorry

end min_value_on_circle_l3330_333056


namespace least_x_for_even_prime_fraction_l3330_333016

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Prime p → 
  Prime (x / (12 * p)) → 
  Even (x / (12 * p)) → 
  (∀ y : ℕ, y > 0 → Prime p → Prime (y / (12 * p)) → Even (y / (12 * p)) → x ≤ y) → 
  x = 48 :=
sorry

end least_x_for_even_prime_fraction_l3330_333016


namespace oliver_stickers_l3330_333023

theorem oliver_stickers (S : ℕ) : 
  (3/5 : ℚ) * (2/3 : ℚ) * S = 54 → S = 135 := by
sorry

end oliver_stickers_l3330_333023


namespace dog_count_l3330_333078

theorem dog_count (dogs people : ℕ) : 
  (4 * dogs + 2 * people = 2 * (dogs + people) + 28) → dogs = 14 := by
  sorry

end dog_count_l3330_333078


namespace tourist_speeds_l3330_333077

theorem tourist_speeds (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (20 / x + 2.5 = 20 / y) ∧ (20 / (x - 2) = 20 / (1.5 * y)) → x = 8 ∧ y = 4 := by
  sorry

end tourist_speeds_l3330_333077


namespace fraction_equation_solution_l3330_333079

theorem fraction_equation_solution (x : ℚ) :
  (1 / (x + 2) + 2 / (x + 2) + x / (x + 2) + 3 / (x + 2) = 4) → x = -2/3 := by
  sorry

end fraction_equation_solution_l3330_333079


namespace farm_animal_leg_difference_l3330_333091

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 6

theorem farm_animal_leg_difference 
  (num_chickens : ℕ) 
  (total_legs : ℕ) 
  (h1 : total_legs = cow_legs * num_cows + chicken_legs * num_chickens)
  (h2 : total_legs > 2 * (num_cows + num_chickens)) :
  total_legs - 2 * (num_cows + num_chickens) = 12 := by
  sorry

end farm_animal_leg_difference_l3330_333091
